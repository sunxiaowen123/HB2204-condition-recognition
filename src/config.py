# -*- coding: utf-8 -*-
"""config.py  HB2204 犁耕工况辨识——全局配置（可复现）
数据：同机同一地块7段采集，200Hz，合成后做整体工况辨识。
标签体系（数据核查后确定，5类）：
  0 Idle/Stop 怠速停车；1 Headland 转弯/轻载机动；2 Transition 入土出土过渡；
  3 MedHeavy 中重负荷犁耕；4 Heavy 重负荷犁耕。
铁律：发电机/驱动电机/电池功率等动力链"果侧"量只做后验，绝不进聚类与分类特征。
"""
import os
import numpy as np

# 仓库自包含：src 的上一级即仓库根，原始数据放在仓库内 data/（发布到 GitHub 后克隆即可复现）
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(PROJ, "data")   # 7 个 plowing_*_processed.csv 与 plowing_lati_longi.mat 所在目录
OUT  = PROJ
DATA_OUT = os.path.join(PROJ, "cache")
os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "main_text"), exist_ok=True)
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)

SEED = 42
np.random.seed(SEED)

# ---- 采样与窗口 ----
FS = 200.0
WIN_S = 3.0
STEP_S = 1.5
WIN_N = int(WIN_S*FS)     # 600
STEP_N = int(STEP_S*FS)   # 300
DS_N = 60                 # 深度模型降采样序列点数(3s@20Hz)

# ---- 通道 ----
# 因侧（前置可用）
FX_COLS   = ["Fx1", "Fx3", "Fx_left_sum"]           # 六分力牵引阻力(Full sensor)
V_GNSS    = "GnssSpeed_mps"
WHEEL_V   = ["v_LF", "v_LR", "v_RR"]               # 轮缘速度
SLIP_COLS = ["Slip_LF", "Slip_LR", "Slip_RR"]       # 左前/左后/右后 滑转率
# 果侧（仅后验）
POWER_COLS = ["P_gen_kW", "P_mot_kW", "P_bat_kW", "SOC"]

# ---- 工况类别 ----
LABELS = ["Idle/Stop", "Headland", "Transition", "MedHeavy", "Heavy"]
LABEL_CN = ["怠速停车", "转弯机动", "过渡", "中重犁耕", "重犁耕"]
N_STEADY = 4   # 稳态分量数（过渡由后验熵单独识别）
CONF_THRESH = 0.85   # GMM最大后验低于此值 -> 过渡候选
DFX_TRANS = 4.0      # kN，相邻窗Fx均值跳变阈值 -> 过渡
# 物理参考阈值（仅用于核查PC-GMM结果，不用于训练标签）
REF = dict(v_idle=0.1, fx_headland=27.0, fx_heavy=35.0)

# ---- 物理参考分档（数据自洽5类，用于与PC-GMM无监督结果对照）----
def reference_label(v, fx, dfx_prev):
    if dfx_prev is not None and abs(dfx_prev) > DFX_TRANS:
        return 2
    if v < REF["v_idle"]:
        return 0
    if fx < REF["fx_headland"]:
        return 1
    if fx < REF["fx_heavy"]:
        return 3
    return 4
