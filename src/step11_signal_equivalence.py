# -*- coding: utf-8 -*-
"""step11_signal_equivalence.py  信号来源等效性核查（支撑"非特殊试验车依赖"）
量产可得: TCU合成车速 WhlBasVehSpd(CAN)、右后编码器轮速 v_RR(CAN)、GPS地速
研究级: 六分力(左前/左后轮速、牵引力)、GPS
核查: 1) CAN车速 vs GPS车速; 2) CAN轮速(RR编码器)+GPS 算出的滑转 vs 六分力滑转(LF/LR)
"""
import os, glob, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from config import ROOT, OUT
files=sorted(glob.glob(os.path.join(ROOT,"plowing_*_processed.csv")),
             key=lambda p:int(os.path.basename(p).split('_')[1]))
df=pd.concat([pd.read_csv(f).assign(seg=int(os.path.basename(f).split('_')[1])) for f in files],ignore_index=True)
df.loc[df.SOC<-999999,'SOC']=np.nan

# 1) CAN车速(TCU) vs GPS车速（作业段 v>0.3）
m=(df.GnssSpeed_mps>0.3)&(df.WhlBasVehSpd_mps>0.3)&(df.Fx_left_sum/1000>27)
g=df.GnssSpeed_mps[m]; t=df.WhlBasVehSpd_mps[m]
r=np.corrcoef(g,t)[0,1]; rmse=np.sqrt(np.mean((g-t)**2)); bias=np.mean(t-g)
print(f"[车速] CAN(TCU) vs GPS: n={m.sum()} r={r:.3f} RMSE={rmse:.3f} m/s bias(TCU-GPS)={bias:.3f}")

# 2) 滑转率：RR(CAN编码器+GPS) vs LF/LR(六分力轮速+GPS)，稳态作业段
w=df[(df.Fx_left_sum/1000>27)&(df.GnssSpeed_mps>1)]
sl=pd.DataFrame({'LF_sixaxis':w.Slip_LF,'LR_sixaxis':w.Slip_LR,'RR_CANencoder':w.Slip_RR}).dropna()
desc=sl.agg(['mean','std']).round(2)
print(f"\n[滑转] 稳态作业段 n={len(sl)}")
print(desc.to_string())
print("\n各轮滑转相关矩阵:"); print(sl.corr().round(3).to_string())

# 图
fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].hexbin(g,t,gridsize=50,cmap='Blues'); lim=[0,2.5]
ax[0].plot(lim,lim,'r--'); ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel('GPS ground speed (m/s)'); ax[0].set_ylabel('CAN/TCU vehicle speed (m/s)')
ax[0].set_title(f'(a) CAN speed vs GPS speed (r={r:.3f}, RMSE={rmse:.2f})')
bp=ax[1].boxplot([sl[c].dropna() for c in sl.columns],tick_labels=sl.columns,showfliers=False,patch_artist=True)
for b,c in zip(bp['boxes'],['#1f77b4','#5dade2','#27ae60']): b.set_facecolor(c)
ax[1].set_ylabel('Wheel slip (%)'); ax[1].set_title('(b) Slip: six-axis (LF/LR) vs CAN-encoder (RR)')
ax[1].grid(axis='y',alpha=.3)
fig.tight_layout(); out=os.path.join(OUT,"supplementary","fig_signal_equivalence.png")
os.makedirs(os.path.dirname(out),exist_ok=True); fig.savefig(out,dpi=140)
pd.DataFrame(dict(r_speed=r,RMSE_speed=rmse,bias_speed=bias,
   slip_LF_mean=sl['LF_sixaxis'].mean(),slip_LR_mean=sl['LR_sixaxis'].mean(),
   slip_RR_mean=sl['RR_CANencoder'].mean()),index=[0]).round(3).to_csv(
   os.path.join(OUT,"results","signal_equivalence.csv"),index=False)
print("\nsaved",out)
