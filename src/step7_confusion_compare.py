# -*- coding: utf-8 -*-
"""17_confusion.py  模型对比总表 + 主模型(XGB)聚合OOF混淆矩阵"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
sys.path.insert(0, os.path.dirname(__file__))
from config import OUT, LABELS, LABEL_CN
RES=os.path.join(OUT,"results")
C=pd.read_csv(os.path.join(RES,"classical_results.csv"))
D=pd.read_csv(os.path.join(RES,"deep_results.csv"))
allr=pd.concat([C,D],ignore_index=True)
order=['RF','SVM','XGB','MLP','LSTM','TCN']
allr['Model']=pd.Categorical(allr.Model,categories=order,ordered=True)
tab=allr.pivot_table(index='Model',columns='FeatureSet',values='MacroF1',observed=True)
acc=allr.pivot_table(index='Model',columns='FeatureSet',values='Acc',observed=True)
print("=== macro-F1 / Acc ===")
print(pd.DataFrame({'F1_CAN':tab['CAN_only'],'F1_Full':tab['Full_sensor'],
                    'Acc_CAN':acc['CAN_only'],'Acc_Full':acc['Full_sensor']}).round(3).to_string())
allr.to_csv(os.path.join(RES,"all_models_summary.csv"),index=False)

oc=np.load(os.path.join(RES,"oof_classical.npz"))
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
for a,key,title in [(ax[0],'XGB_CAN_only','(a) XGBoost, CAN-only (production signals)'),
                    (ax[1],'XGB_Full_sensor','(b) XGBoost, Full sensor (incl. draft force)')]:
    pred=oc[key]; cm=confusion_matrix(oc['y'],pred,labels=range(5))
    cmn=cm/cm.sum(1,keepdims=True)
    im=a.imshow(cmn,cmap='Blues',vmin=0,vmax=1)
    a.set_xticks(range(5),LABELS,rotation=40,ha='right',fontsize=9)
    a.set_yticks(range(5),LABELS,fontsize=9); a.set_title(title,fontsize=11)
    a.set_xlabel('Predicted'); a.set_ylabel('True')
    for i in range(5):
        for j in range(5):
            a.text(j,i,f"{cmn[i,j]*100:.0f}\n({cm[i,j]})" if cm[i,j] else '.',
                   ha='center',va='center',fontsize=8,
                   color='white' if cmn[i,j]>0.5 else 'black')
fig.colorbar(im,ax=ax,shrink=.8,label='Recall')
fig.tight_layout(); p=os.path.join(OUT,"main_text","fig_confusion.png"); os.makedirs(os.path.dirname(p),exist_ok=True)
fig.savefig(p,dpi=150,bbox_inches='tight'); print("saved",p)

# 模型对比柱状图
fig2,b=plt.subplots(figsize=(9,5))
x=np.arange(len(order)); w=.38
b.bar(x-w/2,[tab.loc[m,'CAN_only'] for m in order],w,label='CAN-only (19 feat.)',color='#7fb3d5')
b.bar(x+w/2,[tab.loc[m,'Full_sensor'] for m in order],w,label='Full sensor (31 feat.)',color='#1a5276')
b.set_xticks(x,order); b.set_ylabel('Macro-F1 (LOSO)'); b.set_ylim(0.6,1.0)
b.set_title('Operating-condition recognition: 6 models × 2 feature sets (LOSO 7-fold)')
b.legend(); b.grid(axis='y',alpha=.3)
for i,m in enumerate(order):
    b.text(i-w/2,tab.loc[m,'CAN_only']+.005,f"{tab.loc[m,'CAN_only']:.3f}",ha='center',fontsize=8)
    b.text(i+w/2,tab.loc[m,'Full_sensor']+.005,f"{tab.loc[m,'Full_sensor']:.3f}",ha='center',fontsize=8)
fig2.tight_layout(); p2=os.path.join(OUT,"main_text","fig_model_compare.png"); fig2.savefig(p2,dpi=150)
print("saved",p2)
