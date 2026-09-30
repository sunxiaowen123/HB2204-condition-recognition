# -*- coding: utf-8 -*-
"""resize_geo.py 缩小工作经纬度说明图到可读尺寸"""
import os
from PIL import Image
src = r"D:\Desktop\中国一拖\小论文1\犁耕数据HB2204_新疆\工作经纬度说明.png"
outdir = r"D:\Desktop\中国一拖\小论文1\工况辨识_SCI项目\analysis\ref_docs"
os.makedirs(outdir, exist_ok=True)
img = Image.open(src); w, h = img.size; print("orig", w, h)
scale = 3800 / max(w, h)
nw, nh = int(w*scale), int(h*scale)
img.resize((nw, nh), Image.LANCZOS).save(os.path.join(outdir, "geo_small.png"))
print("small", nw, nh)
# 同时切成左右两半，保证文字清晰
img2 = img.resize((nw, nh), Image.LANCZOS)
half = nw//2
img2.crop((0,0,half+80,nh)).save(os.path.join(outdir,"geo_left.png"))
img2.crop((half-80,0,nw,nh)).save(os.path.join(outdir,"geo_right.png"))
print("done")
