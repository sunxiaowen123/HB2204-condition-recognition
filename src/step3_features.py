# -*- coding: utf-8 -*-
"""13_features.py  因侧特征工程（时域统计 + 频域 + 滑转率）
铁律：功率/电池/电机发电机量不进特征，仅后验。
特征组（支持递进消融）：
  G_V 车速运动学; G_WHEEL 轮速波动; G_SLIP_F 前轮滑转; G_SLIP_R 后轮滑转;
  G_FREQ_CAN 车速频域; G_FX 六分力牵引阻力时域(Full sensor); G_FREQ_FX 阻力频域; G_CPL 力速耦合
输出 feature_matrix.csv（含seg/t0/label）与 feature_groups.json
"""
import os, glob, sys, json
import numpy as np, pandas as pd
from scipy import signal, stats
sys.path.insert(0, os.path.dirname(__file__))
from config import ROOT, DATA_OUT, FS, WIN_N, STEP_N, SLIP_COLS

def freq_feats(x, fs=FS):
    """Hanning窗FFT：低/高频带能量占比、主频、谱熵（0.1-10Hz）"""
    x = np.asarray(x, float); x = x - x.mean()
    if np.allclose(x, 0): return [0.,0.,0.,0.]
    n=len(x); w=np.hanning(n); X=np.abs(np.fft.rfft(x*w))**2
    f=np.fft.rfftfreq(n,1/fs); m=(f>=0.1)&(f<=10)
    ff,PP=f[m],X[m]; P=PP/ (PP.sum()+1e-12)
    lf=PP[(ff>=0.1)&(ff<2)].sum()/(PP.sum()+1e-12)
    hf=PP[(ff>=2)&(ff<=10)].sum()/(PP.sum()+1e-12)
    dom=ff[np.argmax(PP)]
    ent=-(P*np.log(P+1e-12)).sum()/np.log(len(P))
    return [lf,hf,dom,ent]

files=sorted(glob.glob(os.path.join(ROOT,"plowing_*_processed.csv")),
             key=lambda p:int(os.path.basename(p).split('_')[1]))
rows=[]
for f in files:
    seg=int(os.path.basename(f).split('_')[1]); df=pd.read_csv(f); df.loc[df.SOC<-999999,'SOC']=np.nan
    for s in range(0,len(df)-WIN_N+1,STEP_N):
        w=df.iloc[s:s+WIN_N]
        fx=w.Fx_left_sum.values/1000.0; v=w.GnssSpeed_mps.values
        a=np.gradient(v,1/FS)
        d=dict(seg=seg,t0=w.t_s.iloc[0])
        # G_V 车速运动学
        d.update(v_mean=v.mean(),v_std=v.std(ddof=1),v_max=np.percentile(v,95),
                 a_std=a.std(ddof=1),a_p95=np.percentile(np.abs(a),95))
        # G_WHEEL 轮缘速度波动
        for c in ['v_LF','v_LR','v_RR']: d[c+'_std']=np.nanstd(w[c].values)
        # G_SLIP 滑转率（低速v<0.3物理无定义->NaN）
        valid = v.mean()>=0.3
        for c in SLIP_COLS:
            x=w[c].values.astype(float)
            if valid and np.isfinite(x).any():
                d[c+'_mean']=np.nanmean(x); d[c+'_std']=np.nanstd(x)
            else:
                d[c+'_mean']=np.nan; d[c+'_std']=np.nan
        d['slip_cov']=np.isfinite(w[SLIP_COLS].values).mean() if valid else 0.0
        # G_FREQ_CAN 车速频域
        fv=freq_feats(v); d.update(v_lf=fv[0],v_hf=fv[1],v_domf=fv[2],v_ent=fv[3])
        # G_FX 六分力阻力时域（Full sensor）
        d.update(Fx_mean=fx.mean(),Fx_std=fx.std(ddof=1),
                 Fx_cv=fx.std(ddof=1)/(abs(fx.mean())+1e-6),Fx_range=fx.max()-fx.min(),
                 Fx_skew=stats.skew(fx),Fx_kurt=stats.kurtosis(fx),
                 dFx_rms=np.sqrt(np.mean(np.gradient(fx,1/FS)**2)))
        # G_FREQ_FX 阻力频域
        ff=freq_feats(fx); d.update(fx_lf=ff[0],fx_hf=ff[1],fx_domf=ff[2],fx_ent=ff[3])
        # G_CPL 力速耦合
        if fx.std()>1e-6 and v.std()>1e-6:
            d['rho_fxv']=np.corrcoef(fx,v)[0,1]
        else: d['rho_fxv']=0.0
        rows.append(d)
F=pd.DataFrame(rows)

groups={
 'G_V':['v_mean','v_std','v_max','a_std','a_p95'],
 'G_WHEEL':['v_LF_std','v_LR_std','v_RR_std'],
 'G_SLIP_F':['Slip_LF_mean','Slip_LF_std'],
 'G_SLIP_R':['Slip_LR_mean','Slip_RR_mean','Slip_LR_std','Slip_RR_std'],
 'G_FREQ_CAN':['v_lf','v_hf','v_domf','v_ent'],
 'G_FX':['Fx_mean','Fx_std','Fx_cv','Fx_range','Fx_skew','Fx_kurt','dFx_rms'],
 'G_FREQ_FX':['fx_lf','fx_hf','fx_domf','fx_ent'],
 'G_CPL':['rho_fxv'],
}
# 合并标签
lab=pd.read_csv(os.path.join(DATA_OUT,"window_labels.csv"))[['seg','t0','pcgmm']]
F=F.merge(lab,on=['seg','t0'],how='left').rename(columns={'pcgmm':'label'})
F.to_csv(os.path.join(DATA_OUT,"feature_matrix.csv"),index=False)
json.dump(groups,open(os.path.join(DATA_OUT,"feature_groups.json"),'w'),ensure_ascii=False,indent=1)
print("特征矩阵:",F.shape)
print("\n特征组:")
for k,val in groups.items(): print(f"  {k}: {len(val)}维 {val}")
# 递进特征集
sets={
 'CAN_min(G_V)':groups['G_V'],
 '+wheel':groups['G_V']+groups['G_WHEEL'],
 '+slip_front':groups['G_V']+groups['G_WHEEL']+groups['G_SLIP_F'],
 '+slip_rear(CAN_full)':groups['G_V']+groups['G_WHEEL']+groups['G_SLIP_F']+groups['G_SLIP_R']+groups['G_FREQ_CAN']+['slip_cov'],
 'Full_sensor(+Fx)':groups['G_V']+groups['G_WHEEL']+groups['G_SLIP_F']+groups['G_SLIP_R']+groups['G_FREQ_CAN']+['slip_cov']+groups['G_FX']+groups['G_FREQ_FX']+groups['G_CPL'],
}
print("\n特征集维度:")
for k,val in sets.items(): print(f"  {k}: {len(val)}维")
print("\n标签分布:",F.label.value_counts().sort_index().to_dict())
