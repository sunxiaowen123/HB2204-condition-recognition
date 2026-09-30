# -*- coding: utf-8 -*-
"""extract_pdfs.py 提取整机参数与六分力说明书：文本 + 页面渲染图"""
import os, sys
outdir = r"D:\Desktop\中国一拖\小论文1\工况辨识_SCI项目\analysis\ref_docs"
os.makedirs(outdir, exist_ok=True)
try:
    import fitz
except Exception as e:
    print("NO_FITZ", e); sys.exit(1)
pdfs = [r"D:\Desktop\中国一拖\小论文1\犁耕数据HB2204_新疆\整机参数.pdf",
        r"D:\Desktop\中国一拖\小论文1\犁耕数据HB2204_新疆\大马力拖拉机六分力传感器一页说明书.pdf"]
for p in pdfs:
    base = os.path.splitext(os.path.basename(p))[0]
    doc = fitz.open(p)
    print("=====", base, "pages=", len(doc))
    for i, page in enumerate(doc):
        print(f"--- page {i+1} text ---")
        print(page.get_text())
        pix = page.get_pixmap(dpi=170)
        png = os.path.join(outdir, f"{base}_p{i+1}.png")
        pix.save(png); print("rendered", png)
