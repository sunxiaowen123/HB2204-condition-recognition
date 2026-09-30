# -*- coding: utf-8 -*-
"""18_pca.py  PCA机理解析：碎石图 + PC散点(按工况) + 载荷热图"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_OUT, OUT, LABELS
F=pd.read_csv(os.path.join(DATA_OUT,"feature_matrix.csv"))
gpd=json.load(open(os.path.join(DATA_OUT,"feature_groups.json"),encoding='utf-8'))
CAN=gpd['G_V']+gpd['G_WHEEL']+gpd['G_SLIP_F']+gpd['G_SLIP_R']+gpd['G_FREQ_CAN']+['slip_cov']
FULL=CAN+gpd['G_FX']+gpd['G_FREQ_FX']+gpd['G_CPL']
X=SimpleImputer(strategy='median').fit_transform(F[FULL])
X=StandardScaler().fit_transform(X); y=F.label.values
p=PCA(n_components=12).fit(X); Z=p.transform(X)
ev=p.explained_variance_ratio_
print("前6主成分方差解释:",np.round(ev[:6],3),"累计:",np.round(ev.cumsum()[:6],3))

colors={0:'#999999',1:'#2ca02c',2:'#ff7f0e',3:'#1f77b4',4:'#d62728'}
fig,ax=plt.subplots(1,3,figsize=(19,5.2),gridspec_kw={'width_ratios':[1,1.2,1.5]})
ax[0].bar(range(1,11),ev[:10]*100,color='#5499c7'); ax[0].plot(range(1,11),ev.cumsum()[:10]*100,'ro-')
for i in range(10): ax[0].text(i+1,ev[i]*100+.5,f"{ev[i]*100:.0f}",ha='center',fontsize=7)
ax[0].set_xlabel('Principal component'); ax[0].set_ylabel('Variance explained (%)'); ax[0].set_title('(a) Scree plot')
for k in range(5):
    m=y==k; ax[1].scatter(Z[m,0],Z[m,1],s=8,alpha=.4,color=colors[k],label=LABELS[k])
ax[1].set_xlabel(f'PC1 ({ev[0]*100:.0f}%)'); ax[1].set_ylabel(f'PC2 ({ev[1]*100:.0f}%)')
ax[1].set_title('(b) PC1–PC2 by condition'); ax[1].legend(fontsize=8)
# 载荷热图(前4 PC × 关键特征)
key=gpd['G_V'][:3]+['a_std']+gpd['G_SLIP_F'][:1]+['Slip_LR_mean','Slip_RR_mean']+gpd['G_FX'][:4]+['fx_lf']
load=pd.DataFrame(p.components_[:4].T,index=FULL,columns=['PC1','PC2','PC3','PC4']).loc[key]
im=ax[2].imshow(load.values,cmap='RdBu_r',vmin=-.8,vmax=.8,aspect='auto')
ax[2].set_xticks(range(4),['PC1','PC2','PC3','PC4']); ax[2].set_yticks(range(len(key)),key,fontsize=8)
for i in range(len(key)):
    for j in range(4): ax[2].text(j,i,f"{load.values[i,j]:.2f}",ha='center',va='center',fontsize=7)
ax[2].set_title('(c) Loadings (key features)'); fig.colorbar(im,ax=ax[2],shrink=.8)
fig.tight_layout(); out=os.path.join(OUT,"main_text","fig_pca.png"); fig.savefig(out,dpi=150,bbox_inches='tight')
print("saved",out)
print("\nPC1载荷最大特征(按绝对值):")
ld=pd.Series(p.components_[0],index=FULL).abs().sort_values(ascending=False)
print(ld.head(8).round(3).to_string())
