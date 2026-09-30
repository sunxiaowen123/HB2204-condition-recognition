# -*- coding: utf-8 -*-
"""11_build_windows.py  加载7段数据，组内切3s窗(50%重叠)，构建窗口数据集
输出:
  cache/window_stats.csv  每窗统计量(因侧特征原料 + 果侧后验量 + 物理参考标签)
  cache/window_series.npz 每窗降采样序列(60点@20Hz)，供TCN/LSTM
"""
import os, glob, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from config import (ROOT, DATA_OUT, FS, WIN_N, STEP_N, DS_N, SLIP_COLS,
                    WHEEL_V, POWER_COLS, DFX_TRANS, reference_label)

def nanstd(x):
    return np.nanstd(x) if np.isfinite(x).any() else np.nan

files = sorted(glob.glob(os.path.join(ROOT, "plowing_*_processed.csv")),
               key=lambda p: int(os.path.basename(p).split('_')[1]))
stats, series = [], []
for f in files:
    seg = int(os.path.basename(f).split('_')[1])
    df = pd.read_csv(f)
    df.loc[df.SOC < -999999, 'SOC'] = np.nan          # mat3 的 -1e6 坏值
    prev_fx = None
    for s in range(0, len(df)-WIN_N+1, STEP_N):
        w = df.iloc[s:s+WIN_N]
        fx = w.Fx_left_sum.values/1000.0               # kN
        v  = w.GnssSpeed_mps.values
        d = dict(seg=seg, t0=w.t_s.iloc[0],
                 Fx_mean=fx.mean(), Fx_std=fx.std(ddof=1),
                 Fx_min=fx.min(), Fx_max=fx.max(),
                 Fx1_mean=np.abs(w.Fx1.values).mean()/1000.0,
                 Fx3_mean=np.abs(w.Fx3.values).mean()/1000.0,
                 v_mean=v.mean(), v_std=v.std(ddof=1), v_max=v.max())
        # 加速度(运动学)
        a = np.gradient(v, 1/FS)
        d['a_std'], d['a_max'] = a.std(ddof=1), np.percentile(np.abs(a),95)
        # 轮缘速度波动
        for c in WHEEL_V:
            d[c+'_std'] = np.nanstd(w[c].values)
        # 滑转率统计 + 覆盖率
        for c in SLIP_COLS:
            x = w[c].values.astype(float)
            d[c+'_mean'] = np.nanmean(x) if np.isfinite(x).any() else np.nan
            d[c+'_std']  = nanstd(x)
            d[c+'_cov']  = np.isfinite(x).mean()
        # 果侧(仅后验)
        for c in POWER_COLS:
            d[c] = w[c].mean()
        # 物理参考标签
        d['ref'] = reference_label(v.mean(), fx.mean(),
                                   None if prev_fx is None else fx.mean()-prev_fx)
        prev_fx = fx.mean()
        stats.append(d)
        # 降采样序列(60点)：Fx, v, 三轮滑转(NaN->0并记录)
        idx = np.linspace(0, WIN_N-1, DS_N).astype(int)
        sl = np.zeros((DS_N,3))
        for j,c in enumerate(SLIP_COLS):
            x = w[c].values.astype(float); x = np.where(np.isfinite(x), x, 0.0)
            sl[:,j] = x[idx]
        series.append(np.column_stack([fx[idx], v[idx], sl]))
S = pd.DataFrame(stats)
S['dFx'] = S.Fx_mean.diff().abs().fillna(0)
S.loc[S.dFx > DFX_TRANS, 'ref'] = 2
S.to_csv(os.path.join(DATA_OUT,"window_stats.csv"), index=False)
np.savez_compressed(os.path.join(DATA_OUT,"window_series.npz"),
                    seq=np.array(series,dtype=np.float32))
print("窗口总数:", len(S))
print(S.groupby('ref').size().rename({i:n for i,n in enumerate(
      ['Idle','Headland','Transition','MedHeavy','Heavy'])}).to_string())
print("\n各段窗口数:", S.groupby('seg').size().to_dict())
print("已保存 window_stats.csv / window_series.npz ->", DATA_OUT)
