# -*- coding: utf-8 -*-
"""16_ablation.py  递进特征消融 + 滑转轮位对比（XGBoost, LOSO-7折）
让数据裁决：前轮vs后轮滑转、频域、六分力各自的增益（不预设方向）。
"""
import os, sys, json
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_OUT, OUT, LABELS, SEED
RES=os.path.join(OUT,"results")
F=pd.read_csv(os.path.join(DATA_OUT,"feature_matrix.csv"))
gpd=json.load(open(os.path.join(DATA_OUT,"feature_groups.json"),encoding='utf-8'))
y=F.label.values; seg=F.seg.values; segs=sorted(F.seg.unique())

def loso(cols):
    X=F[cols].values; pred=np.full(len(F),-1)
    for ts in segs:
        tr,te=seg!=ts,seg==ts
        imp=SimpleImputer(strategy='median'); sc=StandardScaler()
        Xtr=sc.fit_transform(imp.fit_transform(X[tr])); Xte=sc.transform(imp.transform(X[te]))
        m=XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,subsample=0.9,
                        colsample_bytree=0.9,random_state=SEED,n_jobs=-1,eval_metric='mlogloss')
        m.fit(Xtr,y[tr],sample_weight=compute_sample_weight('balanced',y[tr]))
        pred[te]=m.predict(Xte)
    return accuracy_score(y,pred), f1_score(y,pred,average='macro'), f1_score(y,pred,average=None,labels=range(5))

# ---- 递进消融 ----
prog=[
 ('A0 车速运动学', gpd['G_V']),
 ('A1 +轮速波动', gpd['G_V']+gpd['G_WHEEL']),
 ('A2 +前轮滑转LF', gpd['G_V']+gpd['G_WHEEL']+gpd['G_SLIP_F']),
 ('A3 +后轮滑转LR/RR', gpd['G_V']+gpd['G_WHEEL']+gpd['G_SLIP_F']+gpd['G_SLIP_R']),
 ('A4 +车速频域(CAN_full)', gpd['G_V']+gpd['G_WHEEL']+gpd['G_SLIP_F']+gpd['G_SLIP_R']+gpd['G_FREQ_CAN']+['slip_cov']),
 ('A5 +Fx时域', gpd['G_V']+gpd['G_WHEEL']+gpd['G_SLIP_F']+gpd['G_SLIP_R']+gpd['G_FREQ_CAN']+['slip_cov']+gpd['G_FX']),
 ('A6 +Fx频域', gpd['G_V']+gpd['G_WHEEL']+gpd['G_SLIP_F']+gpd['G_SLIP_R']+gpd['G_FREQ_CAN']+['slip_cov']+gpd['G_FX']+gpd['G_FREQ_FX']),
 ('A7 +力速耦合(Full)', gpd['G_V']+gpd['G_WHEEL']+gpd['G_SLIP_F']+gpd['G_SLIP_R']+gpd['G_FREQ_CAN']+['slip_cov']+gpd['G_FX']+gpd['G_FREQ_FX']+gpd['G_CPL']),
]
rows=[]
for name,cols in prog:
    acc,mf1,pc=loso(cols)
    rows.append(dict(Config=name,nFeat=len(cols),Acc=acc,MacroF1=mf1,**{f"F1_{i}":pc[i] for i in range(5)}))
    print(f"{name:24s} {len(cols):2d}维 Acc={acc:.3f} macroF1={mf1:.3f} 各级F1="+np.array2string(pc,precision=2),flush=True)
R=pd.DataFrame(rows); R.to_csv(os.path.join(RES,"ablation_progressive.csv"),index=False)

# ---- 滑转轮位对比（在车速+轮速基线上）----
base=gpd['G_V']+gpd['G_WHEEL']
wheel_tests=[
 ('基线(无滑转)',base),
 ('+仅前轮LF',base+['Slip_LF_mean','Slip_LF_std']),
 ('+仅左后LR',base+['Slip_LR_mean','Slip_LR_std']),
 ('+仅右后RR',base+['Slip_RR_mean','Slip_RR_std']),
 ('+双后轮LR+RR',base+['Slip_LR_mean','Slip_LR_std','Slip_RR_mean','Slip_RR_std']),
 ('+三轮LF+LR+RR',base+['Slip_LF_mean','Slip_LF_std','Slip_LR_mean','Slip_LR_std','Slip_RR_mean','Slip_RR_std']),
]
wrows=[]
for name,cols in wheel_tests:
    acc,mf1,pc=loso(cols); wrows.append(dict(Config=name,Acc=acc,MacroF1=mf1,**{f"F1_{i}":pc[i] for i in range(5)}))
    print(f"{name:16s} Acc={acc:.3f} macroF1={mf1:.3f}",flush=True)
W=pd.DataFrame(wrows); W.to_csv(os.path.join(RES,"ablation_slip_wheel.csv"),index=False)
print("\nsaved ablation_progressive.csv / ablation_slip_wheel.csv")
