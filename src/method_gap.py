# -*- coding: utf-8 -*-
"""method_gap.py  方法论完备度差距（专家评估，非实测数据）"""
import os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from config import PROJ
items=[
 ("2.1 Platform & field",35,90),
 ("2.2 Acquisition & preprocessing",45,90),
 ("2.3 Hierarchical PC-GMM",40,95),
 ("2.4 Feature engineering",45,92),
 ("2.5 Recognition models",40,90),
 ("2.6 Validation protocol",50,90),
 ("2.7 INT8 & deployment",35,92),
]
items=sorted(items,key=lambda x:x[1])
labels=[i[0] for i in items]; cur=[i[1] for i in items]; tgt=[i[2] for i in items]
import numpy as np
y=np.arange(len(items)); h=0.38
fig,ax=plt.subplots(figsize=(11,5.2))
ax.barh(y+h/2,tgt,height=h,color='#abebc6',label='CEA expected level')
ax.barh(y-h/2,cur,height=h,color='#f1948a',label='Current draft')
for i,(c,t) in enumerate(zip(cur,tgt)):
    ax.text(c+1,i-h/2,str(c),va='center',fontsize=9); ax.text(t+1,i+h/2,str(t),va='center',fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0,105)
ax.set_xlabel('Methodology completeness (0-100, expert assessment)')
ax.set_title('Methodology gap analysis vs CEA reproducibility expectations')
ax.legend(loc='lower right'); fig.tight_layout()
d=os.path.join(PROJ,"analysis"); os.makedirs(d,exist_ok=True)
out=os.path.join(d,"fig_method_gap.png"); fig.savefig(out,dpi=150); print("saved",out)
