# -*- coding: utf-8 -*-
"""step16_pcgmm_sensitivity.py PC-GMM 超参 tau / delta 敏感性"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
sys.path.insert(0, os.path.dirname(__file__)); from config import PROJ
S=pd.read_csv(os.path.join(PROJ,"cache","window_labels.csv"))
fx=S.Fx_mean.values.reshape(-1,1); v=S.v_mean.values.reshape(-1,1)
def fit1d(x,n=2):
    best=None
    for s in range(30):
        g=GaussianMixture(n,n_init=4,random_state=s,reg_covar=1e-3).fit(x)
        mu=np.sort(g.means_.ravel())
        if best is None or g.bic(x)<best.bic(x): best=g
    return best
g0=fit1d(fx,2); o0=np.argsort(g0.means_.ravel()); p0=g0.predict_proba(fx)
post_hi=p0[:,o0[1]]; is_pl=post_hi>p0[:,o0[0]]; maxpost0=np.max(p0,axis=1)
g1=fit1d(v[~is_pl],2); o1=np.argsort(g1.means_.ravel()); pv=g1.predict_proba(v[~is_pl])
base=np.full(len(S),-1,int); sub=np.where(~is_pl)[0]
base[sub[pv[:,o1[0]]>=pv[:,o1[1]]]]=0; base[sub[pv[:,o1[0]]<pv[:,o1[1]]]]=1
pid=np.where(is_pl)[0]; fxp=fx[pid,0]; med=np.median(fxp)
base[pid[fxp<med]]=3; base[pid[fxp>=med]]=4
taus=[.80,.85,.90,.95]; deltas=[3,4,5]; rows=[]
for t in taus:
    for d in deltas:
        lab=base.copy(); lab[(maxpost0<t)|(S.dFx.values>d)]=2
        rows.append(dict(tau=t,delta=d,Agreement=round(accuracy_score(S.ref,lab),3),
                         n_Transition=int((lab==2).sum())))
df=pd.DataFrame(rows); df.to_csv(os.path.join(PROJ,"results","pcgmm_sensitivity.csv"),index=False)
print(df.to_string())
piv=df.pivot(index="tau",columns="delta",values="Agreement")
fig,ax=plt.subplots(1,2,figsize=(12,4.4))
im=ax[0].imshow(piv.values,cmap="Blues",vmin=.90,vmax=.98)
ax[0].set_xticks(range(3),deltas); ax[0].set_yticks(range(4),taus)
ax[0].set_xlabel("delta (kN)"); ax[0].set_ylabel("tau"); ax[0].set_title("Agreement with reference")
for i in range(4):
    for j in range(3): ax[0].text(j,i,f"{piv.values[i,j]:.3f}",ha='center',fontsize=9)
for d in deltas:
    sub=df[df.delta==d]; ax[1].plot(sub.tau,sub.n_Transition,'o-',label=f"delta={d}")
ax[1].set_xlabel("tau"); ax[1].set_ylabel("n Transition windows"); ax[1].legend(); ax[1].set_title("Transition count")
fig.tight_layout(); out=os.path.join(PROJ,"supplementary","fig_pcgmm_sensitivity.png")
fig.savefig(out,dpi=140); print("saved",out)
