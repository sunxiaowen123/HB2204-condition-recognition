# -*- coding: utf-8 -*-
"""14_loso_classical.py  统计学习模型 Leave-One-Segment-Out 交叉验证
模型: RF / SVM-RBF / XGBoost / MLP
特征集: CAN_full(19维,无六分力,量产可得) 与 Full_sensor(31维,含Fx科学上界)
每折内做中位数填充+标准化(不泄漏)；类别不平衡用balanced权重。
输出 outputs/results/classical_results.csv 与OOF预测 oof_classical.npz
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             balanced_accuracy_score)
from xgboost import XGBClassifier
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_OUT, OUT, LABELS, SEED

RES=os.path.join(OUT,"results"); os.makedirs(RES,exist_ok=True)
F=pd.read_csv(os.path.join(DATA_OUT,"feature_matrix.csv"))
groups=json.load(open(os.path.join(DATA_OUT,"feature_groups.json"),encoding='utf-8'))
CAN=(groups['G_V']+groups['G_WHEEL']+groups['G_SLIP_F']+groups['G_SLIP_R']
     +groups['G_FREQ_CAN']+['slip_cov'])
FULL=CAN+groups['G_FX']+groups['G_FREQ_FX']+groups['G_CPL']
SETS={'CAN_only':CAN,'Full_sensor':FULL}

def models():
    return {
     'RF':RandomForestClassifier(n_estimators=400,class_weight='balanced_subsample',
                                 random_state=SEED,n_jobs=-1),
     'SVM':SVC(C=10,gamma='scale',class_weight='balanced',random_state=SEED),
     'XGB':XGBClassifier(n_estimators=400,max_depth=4,learning_rate=0.05,
                         subsample=0.9,colsample_bytree=0.9,random_state=SEED,
                         n_jobs=-1,eval_metric='mlogloss'),
     'MLP':Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),
                     ('net',MLPClassifier(hidden_layer_sizes=(128,64),alpha=1e-3,
                      max_iter=500,early_stopping=True,random_state=SEED))]),
    }

def prep_fit(Xtr,Xte):
    imp=SimpleImputer(strategy='median'); sc=StandardScaler()
    Xtr=sc.fit_transform(imp.fit_transform(Xtr)); Xte=sc.transform(imp.transform(Xte))
    return Xtr,Xte

segs=sorted(F.seg.unique())
oof={}; rows=[]
for setname,cols in SETS.items():
    Xall=F[cols].values; y=F.label.values; g=F.seg.values
    for mname in models():
        oof_pred=np.full(len(F),-1); fold_f1=[]
        for ts in segs:
            tr,te=g!=ts,g==ts
            Xtr,Xte=Xall[tr],Xall[te]; ytr,yte=y[tr],y[te]
            if mname=='MLP':
                m=models()[mname]; m.fit(Xtr,ytr); pred=m.predict(Xte)
            else:
                Xtr2,Xte2=prep_fit(Xtr,Xte)
                m=models()[mname]
                if mname=='XGB':
                    m.fit(Xtr2,ytr,sample_weight=__import__('sklearn').utils.class_weight.compute_sample_weight('balanced',ytr))
                else:
                    m.fit(Xtr2,ytr)
                pred=m.predict(Xte2)
            oof_pred[te]=pred
            if len(np.unique(yte))>1:
                fold_f1.append(f1_score(yte,pred,average='macro'))
        acc=accuracy_score(y,oof_pred); mf1=f1_score(y,oof_pred,average='macro')
        bacc=balanced_accuracy_score(y,oof_pred)
        pc=f1_score(y,oof_pred,average=None,labels=range(5))
        oof[f"{mname}_{setname}"]=oof_pred
        rows.append(dict(Model=mname,FeatureSet=setname,Acc=acc,MacroF1=mf1,BalAcc=bacc,
                         FoldF1_mean=np.mean(fold_f1),FoldF1_std=np.std(fold_f1),
                         **{f"F1_{LABELS[i]}":pc[i] for i in range(5)}))
        print(f"{mname:4s} {setname:11s} Acc={acc:.3f} macroF1={mf1:.3f} balAcc={bacc:.3f} "
              f"foldF1={np.mean(fold_f1):.3f}±{np.std(fold_f1):.3f}")
R=pd.DataFrame(rows); R.to_csv(os.path.join(RES,"classical_results.csv"),index=False)
np.savez(os.path.join(RES,"oof_classical.npz"),**oof,y=F.label.values,seg=F.seg.values)
print("\n已保存 classical_results.csv / oof_classical.npz")
print("\n各类F1（Full_sensor）:")
print(R[R.FeatureSet=='Full_sensor'][['Model']+[f"F1_{l}" for l in LABELS]].round(3).to_string(index=False))
