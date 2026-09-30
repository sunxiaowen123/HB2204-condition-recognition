# -*- coding: utf-8 -*-
"""inspect_cache.py 打印 cache 文件结构，供编写新实验脚本"""
import os, pandas as pd, numpy as np, json
cache = r"D:\Desktop\中国一拖\小论文1\工况辨识_SCI项目\cache"
for f in ["window_stats.csv","window_labels.csv","feature_matrix.csv"]:
    df = pd.read_csv(os.path.join(cache,f))
    print("=====",f,df.shape); print(df.columns.tolist()); print(df.head(2).to_string())
for f in ["oof_classical.npz","oof_deep.npz","window_series.npz"]:
    d=np.load(os.path.join(cache,f), allow_pickle=True)
    print("=====",f,"keys=",d.files)
    for k in d.files:
        try: print("  ",k, d[k].shape, d[k].dtype)
        except: print("  ",k, type(d[k]))
fg=json.load(open(os.path.join(cache,"feature_groups.json"),encoding="utf-8"))
print("feature_groups keys=", list(fg.keys()) if isinstance(fg,dict) else type(fg))
