# -*- coding: utf-8 -*-
"""12_pc_gmm.py  分层物理约束GMM伪标签 (Hierarchical PC-GMM)
第1层 Fx一维2分量: 非犁耕态 vs 犁耕态(双峰谷底,物理清晰)
第2层 非犁耕内按v 2分量 -> Idle/Headland; 犁耕内在Fx上2分量(有序软分档)->MedHeavy/Heavy
过渡 Transition: 第1层后验混合(maxpost<0.85,入土出土穿越谷底) 或 相邻窗Fx跳变>4kN
全部无监督fit；物理参考ref仅用于盲对照，不参与拟合。
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_OUT, OUT, LABELS, CONF_THRESH, DFX_TRANS

S = pd.read_csv(os.path.join(DATA_OUT,"window_stats.csv"))
fx = S.Fx_mean.values.reshape(-1,1); v = S.v_mean.values.reshape(-1,1)
rng = list(range(30))

def fit1d(x, n=2):
    best=None
    for s in rng:
        g=GaussianMixture(n_components=n,n_init=4,random_state=s,reg_covar=1e-3).fit(x)
        order=np.argsort(g.means_.ravel())
        # 物理约束：分量均值需有序且分离
        mu=np.sort(g.means_.ravel())
        sep=(mu[-1]-mu[0])> (np.sqrt(g.covariances_.ravel()[order[0]])+np.sqrt(g.covariances_.ravel()[order[-1]]))*0.5
        if best is None or (sep and g.bic(x)<best['bic']):
            best=dict(g=g,bic=g.bic(x),order=order)
    return best['g'], best['order']

# 第1层：非犁耕 vs 犁耕
g0,o0 = fit1d(fx,2)
p0 = g0.predict_proba(fx)
post_low, post_high = p0[:,o0[0]], p0[:,o0[1]]
is_plough = post_high > post_low
print(f"第1层 Fx分界(后验相等)≈", end=" ")
xs=np.linspace(fx.min(),fx.max(),2000).reshape(-1,1)
pp=g0.predict_proba(xs); cross=xs[np.argmin(abs(pp[:,o0[1]]-pp[:,o0[0]]))][0]
print(f"{cross:.1f} kN  非犁耕{(~is_plough).sum()}窗 / 犁耕{is_plough.sum()}窗")

# 第2层a：非犁耕内 Idle/Headland (按v)
g1,o1 = fit1d(v[~is_plough],2)
lab=np.full(len(S),-1,dtype=int)
pv=g1.predict_proba(v[~is_plough])
sub_idx=np.where(~is_plough)[0]
lab[sub_idx[pv[:,o1[0]].argmax(0)>=0]]= -1  # placeholder
lab[sub_idx[np.where(pv[:,o1[0]]>=pv[:,o1[1]])[0]]]=0
lab[sub_idx[np.where(pv[:,o1[0]]< pv[:,o1[1]])[0]]]=1
print(f"第2层 非犁耕: Idle v*={g1.means_.ravel()[o1[0]]:.2f}, Headland v*={g1.means_.ravel()[o1[1]]:.2f} m/s")

# 第2层b：犁耕内 MedHeavy/Heavy —— 单峰无自然分界，采用数据驱动的"有序负荷分档"
# （以犁耕态Fx中位数为界；论文如实表述为ordinal load grading，而非GMM自然簇）
pidx=np.where(is_plough)[0]
fx_plough=fx[pidx,0]; fx_med=np.median(fx_plough)
lab[pidx[fx_plough < fx_med]]=3
lab[pidx[fx_plough>=fx_med]]=4
print(f"第2层 犁耕有序分档: 阈值=Fx中位数 {fx_med:.1f} kN -> MedHeavy {(fx_plough<fx_med).sum()} / Heavy {(fx_plough>=fx_med).sum()}")

# 过渡：第1层后验混合 或 dFx跳变
maxpost0=np.maximum(post_low,post_high)
trans=(maxpost0<CONF_THRESH)|(S.dFx.values>DFX_TRANS)
lab[trans]=2
S['pcgmm']=lab; S['maxpost0']=maxpost0

# ---------- 盲对照 ----------
acc=accuracy_score(S.ref,S.pcgmm)
cm=confusion_matrix(S.ref,S.pcgmm,labels=[0,1,2,3,4])
sil=silhouette_score(np.c_[S.Fx_mean,S.v_mean], np.where(lab==2,-1,lab))
print(f"\nPC-GMM vs 物理参考 一致率={acc:.3f}  稳态轮廓(过渡除外)={sil:.3f}")
print("混淆矩阵(行参考,列PCGMM; 怠速/转弯/过渡/中重/重):"); print(cm)
print("\nPCGMM各类:",{LABELS[i]:int((S.pcgmm==i).sum()) for i in range(5)})
print("参考各类:",{LABELS[i]:int((S.ref==i).sum()) for i in range(5)})
print("\n各类物理量(PC-GMM):")
print(S.groupby('pcgmm').agg(n=('Fx_mean','size'),Fx=('Fx_mean','mean'),Fxsd=('Fx_mean','std'),
      v=('v_mean','mean'),Pmot=('P_mot_kW','mean'),Pgen=('P_gen_kW','mean'),Pbat=('P_bat_kW','mean'),
      SOC=('SOC','mean'),LF=('Slip_LF_mean','mean'),RR=('Slip_RR_mean','mean')).round(2).to_string())
S.to_csv(os.path.join(DATA_OUT,"window_labels.csv"),index=False)

fig,ax=plt.subplots(1,3,figsize=(18,5)); colors={0:'#999999',1:'#2ca02c',2:'#ff7f0e',3:'#1f77b4',4:'#d62728'}
for k in range(5):
    m=S.pcgmm==k; ax[0].scatter(S.v_mean[m],S.Fx_mean[m],s=8,alpha=.4,color=colors[k],label=LABELS[k])
ax[0].set_title('(a) Hierarchical PC-GMM labels');ax[0].set_xlabel('v (m/s)');ax[0].set_ylabel('Fx (kN)');ax[0].legend(fontsize=8)
for k in range(5):
    m=S.ref==k; ax[1].scatter(S.v_mean[m],S.Fx_mean[m],s=8,alpha=.4,color=colors[k],label=LABELS[k])
ax[1].set_title('(b) Physics reference');ax[1].set_xlabel('v (m/s)');ax[1].set_ylabel('Fx (kN)');ax[1].legend(fontsize=8)
ax[2].imshow(cm,cmap='Blues');ax[2].set_xticks(range(5),[s[:6] for s in LABELS],rotation=45,ha='right',fontsize=8)
ax[2].set_yticks(range(5),[s[:6] for s in LABELS],fontsize=8);ax[2].set_title(f'(c) Agreement {acc:.2%}')
for i in range(5):
    for j in range(5):
        if cm[i,j]:ax[2].text(j,i,cm[i,j],ha='center',va='center',fontsize=8)
ax[2].set_xlabel('PC-GMM');ax[2].set_ylabel('Reference')
fig.tight_layout();p=os.path.join(OUT,"pcgmm_validation.png");fig.savefig(p,dpi=140);print("\nsaved",p)
