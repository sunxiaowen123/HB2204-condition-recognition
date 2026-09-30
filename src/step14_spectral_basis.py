# -*- coding: utf-8 -*-
"""step14_spectral_basis.py 频域带宽依据：Fx/v 的 Welch PSD，标注截止频率"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import signal
sys.path.insert(0, os.path.dirname(__file__)); from config import ROOT, PROJ, FS
fx=[]; v=[]
for f in sorted(glob.glob(os.path.join(ROOT,"plowing_*_processed.csv"))):
    d=pd.read_csv(glob.glob(f)[0])
    x=d["Fx_left_sum"].values
    if np.nanmedian(np.abs(x))>1000: x=x/1000.0   # N -> kN
    fx.append(x); v.append(d["GnssSpeed_mps"].values)
fx=np.concatenate(fx); v=np.concatenate(v)
work=fx>20   # 犁耕段
def welch(x,fs=FS):
    f,p=signal.welch(x,fs=fs,nperseg=int(fs*6),noverlap=int(fs*3))
    return f,p
f_fx,p_fx=welch(np.nan_to_num(fx[work]))
f_v,p_v=welch(np.nan_to_num(v))
fig,ax=plt.subplots(1,3,figsize=(16,4.4))
ax[0].semilogy(f_fx,p_fx+1e-8); ax[0].axvline(5,color='r',ls='--'); ax[0].set_xlim(0,40)
ax[0].set_title("Draft-force PSD (tillage, 0-40 Hz)"); ax[0].set_xlabel("Hz"); ax[0].set_ylabel("PSD")
ax[1].semilogy(f_fx,f_fx*0+np.nan,"w"); ax[1].semilogy(f_fx,p_fx+1e-8); ax[1].axvline(5,color='r',ls='--')
ax[1].set_xlim(0,10); ax[1].set_title("Zoom 0-10 Hz (cutoff 5 Hz)"); ax[1].set_xlabel("Hz")
ax[2].semilogy(f_v,p_v+1e-10); ax[2].set_xlim(0,20); ax[2].set_title("Ground-speed PSD"); ax[2].set_xlabel("Hz")
fig.tight_layout()
out=os.path.join(PROJ,"supplementary","fig_spectral_basis.png"); fig.savefig(out,dpi=150)
frac=np.trapezoid(p_fx[f_fx<=5],f_fx[f_fx<=5])/np.trapezoid(p_fx,f_fx)
print("Fx energy fraction below 5 Hz =",round(float(frac),3)); print("saved",out)
