# -*- coding: utf-8 -*-
"""step12_figure1.py  研究框架流程图 (Figure 1)"""
import os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(__file__))
from config import OUT

stages=[
 ("A. Field measurement",
  ["HB2204 series-hybrid tractor (220 hp)",
   "6-axis force transducer + GNSS, 200 Hz",
   "7 segments, one field, fixed depth 30 cm"],"#d6eaf8"),
 ("B. Preprocessing & windowing",
  ["Hampel + zero-phase Butterworth filtering",
   "3-s windows, 50% overlap",
   "3200 windows"],"#d1f2eb"),
 ("C. Hierarchical PC-GMM",
  ["L1: Fx -> non-tillage / tillage",
   "L2: v -> Idle/Headland; Fx ordinal -> Med/Heavy",
   "Posterior mixture -> Transition"],"#fcf3cf"),
 ("D. Features & recognition",
  ["CAN-only (19) / Full incl. Fx (31)",
   "Time-domain + wheel slip",
   "RF SVM XGB MLP LSTM TCN; LOSO 7-fold"],"#fae5d3"),
 ("E. Validation & deployment",
  ["Confusion matrix, ablation, PCA",
   "Posterior powertrain power check",
   "INT8 quantization -> edge MCU"],"#f5eef8"),
]
fig,ax=plt.subplots(figsize=(20,4.6)); ax.set_xlim(0,100); ax.set_ylim(0,30); ax.axis('off')
w=17.5; gap=2.0; x=1.5; centers=[]
for title,lines,fc in stages:
    box=FancyBboxPatch((x,4),w,22,boxstyle="round,pad=0.3,rounding_size=0.8",
                       fc=fc,ec='#34495e',lw=1.3); ax.add_patch(box)
    ax.text(x+w/2,23.3,title,ha='center',va='center',fontsize=11,fontweight='bold')
    for i,ln in enumerate(lines):
        ax.text(x+w/2,18.5-i*4.3,ln,ha='center',va='center',fontsize=8.6)
    centers.append((x,x+w)); x+=w+gap
for i in range(len(stages)-1):
    x0=centers[i][1]; x1=centers[i+1][0]
    ar=FancyArrowPatch((x0+0.1,15),(x1-0.1,15),arrowstyle='-|>',mutation_scale=22,lw=2,color='#2c3e50')
    ax.add_patch(ar)
ax.text(50,1.2,"Powertrain power (motor/generator/battery, SOC) used ONLY for posterior validation, never for labeling or classification",
        ha='center',fontsize=9,style='italic',color='#922b21')
fig.tight_layout(); out=os.path.join(OUT,"main_text","fig1_flowchart.png")
fig.savefig(out,dpi=160,bbox_inches='tight'); print("saved",out)
