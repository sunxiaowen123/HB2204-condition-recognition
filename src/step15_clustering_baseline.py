# -*- coding: utf-8 -*-
"""step15_clustering_baseline.py 聚类基线对比：KMeans / 无约束2D GMM vs PC-GMM"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, os.path.dirname(__file__)); from config import PROJ
S=pd.read_csv(os.path.join(PROJ,"cache","window_labels.csv"))
X=np.c_[S.Fx_mean.values, S.v_mean.values]; ref=S.ref.values
def hung_acc(yt,yp,k=5):
    cm=np.zeros((k,k),int)
    for t,p in zip(yt,yp): cm[t,p]+=1
    r,c=linear_sum_assignment(-cm); return cm[r,c].sum()/len(yt)
preds={
 "KMeans (2D, K=5)": KMeans(5,n_init=20,random_state=42).fit_predict(X),
 "GMM (2D full, K=5)": GaussianMixture(5,covariance_type="full",n_init=5,random_state=42).fit_predict(X),
 "GMM (2D diag, K=5)": GaussianMixture(5,covariance_type="diag",n_init=5,random_state=42).fit_predict(X),
 "PC-GMM (proposed)": S.pcgmm.values,
}
rows=[]
for name,p in preds.items():
    rows.append(dict(Method=name,ARI=round(adjusted_rand_score(ref,p),3),
                     Accuracy_hungarian=round(hung_acc(ref,p),3)))
df=pd.DataFrame(rows); df.to_csv(os.path.join(PROJ,"results","clustering_baseline.csv"),index=False)
print(df.to_string())
# 物理诊断：2D full GMM 各簇 Fx/v 均值
g=GaussianMixture(5,covariance_type="full",n_init=5,random_state=42).fit(X); pr=g.predict(X)
diag=pd.DataFrame([dict(cluster=c,Fx=round(X[pr==c,0].mean(),1),v=round(X[pr==c,1].mean(),2),n=int((pr==c).sum())) for c in range(5)])
print("\n2D GMM cluster means:\n",diag.to_string())
# 图
fig,ax=plt.subplots(1,3,figsize=(18,5.2))
cols=plt.cm.tab10(np.linspace(0,1,5))
for ax_,(name,p),title in zip(ax,list(preds.items())[0:2]+[("PC-GMM (proposed)",preds["PC-GMM (proposed)"])],
        ["KMeans K=5","Unconstrained 2D GMM K=5","PC-GMM (proposed)"]):
    for c in range(5):
        m=p==c; ax_.scatter(X[m,1],X[m,0],s=8,alpha=.45,color=cols[c])
    ax_.set_title(title); ax_.set_xlabel("v (m/s)"); ax_.set_ylabel("Fx (kN)")
fig.tight_layout(); out=os.path.join(PROJ,"supplementary","fig_clustering_baseline.png")
fig.savefig(out,dpi=140); print("saved",out)
