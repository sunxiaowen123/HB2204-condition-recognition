# -*- coding: utf-8 -*-
"""19_posthoc.py  动力链功率后验物理校验（功率绝不参与辨识）
1) 各类动力链功率统计；2) Fx-v等功率线解释重档降速、功率不严格单调；
3) 控制车速后比较中重/重；4) 串联功率平衡 Pmot≈|Pgen|+Pbat 一致性。
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_OUT, OUT, LABELS
S=pd.read_csv(os.path.join(DATA_OUT,"window_labels.csv"))
S['P_draw']=S.Fx_mean*S.v_mean     # 牵引功率 kW(kN*m/s)
colors={0:'#999999',1:'#2ca02c',2:'#ff7f0e',3:'#1f77b4',4:'#d62728'}

g=S.groupby('pcgmm').agg(n=('Fx_mean','size'),Fx=('Fx_mean','mean'),v=('v_mean','mean'),
   P_draw=('P_draw','mean'),Pmot=('P_mot_kW','mean'),Pgen=('P_gen_kW','mean'),
   Pbat=('P_bat_kW','mean'),SOC=('SOC','mean')).round(2)
print("=== 各类动力链后验（PC-GMM标签）==="); print(g.to_string())

# 控制车速：作业两级在共同车速区间
work=S[S.pcgmm.isin([3,4])]
bins=np.arange(1.2,2.01,0.2); work=work.assign(vb=pd.cut(work.v_mean,bins))
ctrl=work.groupby(['pcgmm','vb'],observed=True).agg(Fx=('Fx_mean','mean'),
      Pmot=('P_mot_kW','mean'),n=('Fx_mean','size')).round(1)
print("\n=== 控制车速后 中重(3)/重(4) ==="); print(ctrl.to_string())

# 功率平衡：Pmot 与 -(Pgen)+Pbat
bal=S[np.abs(S.P_mot_kW)>5]
r=np.corrcoef(bal.P_mot_kW, -bal.P_gen_kW+bal.P_bat_kW)[0,1]
print(f"\n串联功率平衡 corr(Pmot, |Pgen|+Pbat)={r:.3f}  中位供给/需求比="
      f"{np.nanmedian((-bal.P_gen_kW+bal.P_bat_kW)/bal.P_mot_kW):.2f}")

fig,ax=plt.subplots(2,2,figsize=(14,10))
# (a) 功率条形
mm=g[['P_draw','Pmot','Pgen']].values; x=np.arange(5); w=.25
ax[0,0].bar(x-w,mm[:,0],w,label='Drawbar Fx*v',color='#8e44ad')
ax[0,0].bar(x,mm[:,1],w,label='Drive motor Pmot',color='#c0392b')
ax[0,0].bar(x+w,mm[:,2],w,label='Generator Pgen',color='#2471a3')
ax[0,0].axhline(0,color='k',lw=.6); ax[0,0].set_xticks(x,LABELS,rotation=25,fontsize=8)
ax[0,0].set_ylabel('Power (kW)'); ax[0,0].set_title('(a) Powertrain power by condition (posterior)'); ax[0,0].legend(fontsize=8)
# (b) Fx-v 等功率线
for k in range(5):
    m=S.pcgmm==k; ax[0,1].scatter(S.v_mean[m],S.Fx_mean[m],s=8,alpha=.35,color=colors[k],label=LABELS[k])
vv=np.linspace(0.3,2.3,100)
for P in[30,55,80]:
    ax[0,1].plot(vv,P/vv,'--',color='gray',lw=.8); ax[0,1].text(2.25,P/2.25,f'{P}kW',fontsize=7,color='gray')
ax[0,1].set_xlabel('v (m/s)'); ax[0,1].set_ylabel('Fx (kN)'); ax[0,1].legend(fontsize=8)
ax[0,1].set_title('(b) Fx–v with constant-power hyperbolae')
# (c) 控制车速
piv=work.groupby(['vb','pcgmm'],observed=True)['P_mot_kW'].mean().unstack()
piv_f=work.groupby(['vb','pcgmm'],observed=True)['Fx_mean'].mean().unstack()
xb=np.arange(len(piv))
ax[1,0].bar(xb-.18,piv_f[3],.36,color='#1f77b4',label='MedHeavy Fx')
ax[1,0].bar(xb+.18,piv_f[4],.36,color='#d62728',label='Heavy Fx')
ax[1,0].set_xticks(xb,[str(i)[1:-1].replace(',','-') for i in piv.index],fontsize=8)
ax[1,0].set_ylabel('Fx (kN)'); ax[1,0].set_title('(c) At matched speed: Heavy always has higher Fx'); ax[1,0].legend(fontsize=8)
ax2=ax[1,0].twinx(); ax2.plot(xb,piv[3],'b--o',ms=4); ax2.plot(xb,piv[4],'r--o',ms=4); ax2.set_ylabel('Pmot (kW)')
# (d) 功率平衡散点
ax[1,1].scatter(bal.P_mot_kW,-bal.P_gen_kW+bal.P_bat_kW,s=6,alpha=.3,color='#117a65')
lim=[0,200]; ax[1,1].plot(lim,lim,'k--',lw=1)
ax[1,1].set_xlabel('Drive motor Pmot (kW)'); ax[1,1].set_ylabel('Generator+battery supply (kW)')
ax[1,1].set_title(f'(d) Series-hybrid power balance (r={r:.2f})')
fig.tight_layout(); out=os.path.join(OUT,"main_text","fig_posthoc_power.png"); fig.savefig(out,dpi=150)
g.to_csv(os.path.join(OUT,"results","posthoc_power_table.csv"))
print("saved",out)
