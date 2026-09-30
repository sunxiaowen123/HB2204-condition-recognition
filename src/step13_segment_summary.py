# -*- coding: utf-8 -*-
"""step13_segment_summary.py 分段汇总表"""
import os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from config import PROJ
lab=pd.read_csv(os.path.join(PROJ,"cache","window_labels.csv"))
rows=[]
for s,g in lab.groupby("seg"):
    dur=g.shape[0]*1.5/60.0
    vc=g["pcgmm"].value_counts(normalize=True)
    rows.append(dict(Segment=int(s), Windows=g.shape[0], Duration_min=round(dur,2),
        Fx_mean_kN=round(g.Fx_mean.mean(),1), v_mean_ms=round(g.v_mean.mean(),2),
        Idle_pct=round(100*vc.get(0,0),1), Headland_pct=round(100*vc.get(1,0),1),
        Transition_pct=round(100*vc.get(2,0),1), MedHeavy_pct=round(100*vc.get(3,0),1),
        Heavy_pct=round(100*vc.get(4,0),1)))
df=pd.DataFrame(rows)
tot=pd.DataFrame([dict(Segment="ALL",Windows=lab.shape[0],Duration_min=round(lab.shape[0]*1.5/60,1),
    Fx_mean_kN=round(lab.Fx_mean.mean(),1),v_mean_ms=round(lab.v_mean.mean(),2),
    Idle_pct=round(100*(lab.pcgmm==0).mean(),1),Headland_pct=round(100*(lab.pcgmm==1).mean(),1),
    Transition_pct=round(100*(lab.pcgmm==2).mean(),1),MedHeavy_pct=round(100*(lab.pcgmm==3).mean(),1),
    Heavy_pct=round(100*(lab.pcgmm==4).mean(),1))])
df=pd.concat([df,tot],ignore_index=True)
df.to_csv(os.path.join(PROJ,"results","segment_summary.csv"),index=False)
print(df.to_string())
