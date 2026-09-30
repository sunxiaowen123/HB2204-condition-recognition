# -*- coding: utf-8 -*-
"""20_int8.py  工程特征子集 + MLP INT8静态量化(PTQ)边缘部署评估
特征集: ENG(CAN top10量产信号) 与 Full(31维含Fx)；模型FP32 vs INT8(LOSO泛化)
部署: 模型大小、单窗推理时延(CPU单线程)；XGBoost作树模型参照。
"""
import os, sys, json, time, tempfile, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_OUT, OUT, SEED
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(1)
RES=os.path.join(OUT,"results")
F=pd.read_csv(os.path.join(DATA_OUT,"feature_matrix.csv"))
gpd=json.load(open(os.path.join(DATA_OUT,"feature_groups.json"),encoding='utf-8'))
CAN=gpd['G_V']+gpd['G_WHEEL']+gpd['G_SLIP_F']+gpd['G_SLIP_R']+gpd['G_FREQ_CAN']+['slip_cov']
FULL=CAN+gpd['G_FX']+gpd['G_FREQ_FX']+gpd['G_CPL']
y=F.label.values; seg=F.seg.values; segs=sorted(F.seg.unique())

Xc=SimpleImputer(strategy='median').fit_transform(F[CAN])
xb0=XGBClassifier(n_estimators=400,max_depth=4,learning_rate=.05,random_state=SEED,n_jobs=-1,eval_metric='mlogloss')
xb0.fit(Xc,y,sample_weight=compute_sample_weight('balanced',y))
imp=pd.Series(xb0.feature_importances_,index=CAN).sort_values(ascending=False)
ENG=list(imp.head(10).index)
print("工程CAN特征子集(top10):",ENG)
SETS={'ENG(CAN10)':ENG,'Full(31)':FULL}

class QMLP(nn.Module):
    def __init__(s,d,h=64):
        super().__init__(); s.q=torch.quantization.QuantStub()
        s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h//2),nn.ReLU(),nn.Linear(h//2,5))
        s.dq=torch.quantization.DeQuantStub()
    def forward(s,x): return s.dq(s.net(s.q(x)))

def train_fp(Xtr,ytr,d,epochs=120):
    cw=torch.tensor(len(ytr)/(5*np.bincount(ytr,minlength=5)),dtype=torch.float32)
    net=QMLP(d); opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-4); lf=nn.CrossEntropyLoss(weight=cw)
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long); n=len(Xt)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,128):
            idx=perm[i:i+128]; opt.zero_grad(); lf(net(Xt[idx]),yt[idx]).backward(); opt.step()
    return net

def quantize(net,Xcal):
    net.eval(); net.qconfig=torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(net,inplace=True)
    with torch.no_grad(): net(torch.tensor(Xcal,dtype=torch.float32))
    return torch.quantization.convert(net.eval(),inplace=False)

def xgb_loso(cols):
    X=SimpleImputer(strategy='median').fit_transform(F[cols]); pr=np.full(len(F),-1)
    for ts in segs:
        tr,te=seg!=ts,seg==ts; sc=StandardScaler()
        m=XGBClassifier(n_estimators=400,max_depth=4,learning_rate=.05,random_state=SEED,n_jobs=-1,eval_metric='mlogloss')
        m.fit(sc.fit_transform(X[tr]),y[tr],sample_weight=compute_sample_weight('balanced',y[tr]))
        pr[te]=m.predict(sc.transform(X[te]))
    return accuracy_score(y,pr), f1_score(y,pr,average='macro')

results=[]; deploy={}
for setname,cols in SETS.items():
    X=SimpleImputer(strategy='median').fit_transform(F[cols]); d=X.shape[1]
    fp_pred=np.full(len(F),-1); q_pred=np.full(len(F),-1); fp=qn=None
    for ts in segs:
        tr,te=seg!=ts,seg==ts; sc=StandardScaler()
        Xtr=sc.fit_transform(X[tr]); Xte=sc.transform(X[te])
        net=train_fp(Xtr,y[tr],d); net.eval()
        with torch.no_grad(): fp_pred[te]=net(torch.tensor(Xte,dtype=torch.float32)).argmax(1).numpy()
        qnet=quantize(net,Xtr[:500])
        with torch.no_grad(): q_pred[te]=qnet(torch.tensor(Xte,dtype=torch.float32)).argmax(1).numpy()
        fp,qn=net,qnet
    xa,xm=xgb_loso(cols)
    results.append((setname,'XGBoost',xa,xm,np.nan,np.nan))
    results.append((setname,'MLP FP32',accuracy_score(y,fp_pred),f1_score(y,fp_pred,average='macro'),np.nan,np.nan))
    results.append((setname,'MLP INT8',accuracy_score(y,q_pred),f1_score(y,q_pred,average='macro'),np.nan,np.nan))
    print(f"{setname:10s} XGB F1={xm:.3f} | MLP FP32 F1={f1_score(y,fp_pred,average='macro'):.3f} | INT8 F1={f1_score(y,q_pred,average='macro'):.3f}",flush=True)
    if setname.startswith('ENG'): deploy=dict(fp=fp,qn=qn,X=X,cols=cols)

# ---- 部署大小/时延(ENG子集, 全数据最终模型) ----
sc=StandardScaler(); Xs=sc.fit_transform(deploy['X']); d=Xs.shape[1]
fp=train_fp(Xs,y,d); qn=quantize(fp,Xs[:500])
def msize(m):
    f=tempfile.NamedTemporaryFile(suffix='.pt',delete=False); torch.save(m.state_dict(),f); f.close()
    s=os.path.getsize(f.name)/1024; os.unlink(f.name); return s
xt=torch.tensor(Xs[:1],dtype=torch.float32)
def latency(m,n=2000):
    with torch.no_grad():
        for _ in range(100): m(xt)
        t0=time.perf_counter()
        for _ in range(n): m(xt)
    return (time.perf_counter()-t0)/n*1e6
xbs=XGBClassifier(n_estimators=400,max_depth=4,random_state=SEED,n_jobs=1,eval_metric='mlogloss').fit(Xs,y)
fx=os.path.join(RES,'xgb_eng.ubj'); xbs.save_model(fx); xgb_sz=os.path.getsize(fx)/1024
t0=time.perf_counter()
for _ in range(2000): xbs.predict(Xs[:1])
xgb_lat=(time.perf_counter()-t0)/2000*1e6
dep=pd.DataFrame([
 dict(Model='XGBoost (10 CAN feat.)',Size_KB=round(xgb_sz,1),Latency_us=round(xgb_lat,1)),
 dict(Model='MLP FP32 (64-32)',Size_KB=round(msize(fp),1),Latency_us=round(latency(fp),1)),
 dict(Model='MLP INT8 PTQ',Size_KB=round(msize(qn),1),Latency_us=round(latency(qn),1))])
R=pd.DataFrame(results,columns=['FeatureSet','Model','Acc','MacroF1','Size_KB','Latency_us'])
print("\n=== LOSO精度 ==="); print(R.round(3).to_string(index=False))
print("\n=== 部署评估(CPU单线程,单窗) ==="); print(dep.to_string(index=False))
R.to_csv(os.path.join(RES,"int8_accuracy.csv"),index=False)
dep.to_csv(os.path.join(RES,"deployment_int8.csv"),index=False)
pd.DataFrame({'feature':ENG}).to_csv(os.path.join(RES,"engineering_features.csv"),index=False)
print("\nsaved.")
