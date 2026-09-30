# -*- coding: utf-8 -*-
"""step17_significance.py 模型间差异显著性：McNemar(OOF) + 折级配对 t / Wilcoxon"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(__file__)); from config import PROJ
c=np.load(os.path.join(PROJ,"cache","oof_classical.npz")); d=np.load(os.path.join(PROJ,"cache","oof_deep.npz"))
y=c["y"]; seg=c["seg"]
sets={
 "CAN":{k.split("_")[0]:c[k] for k in c.files if k.endswith("CAN_only")},
 "Full":{k.split("_")[0]:c[k] for k in c.files if k.endswith("Full_sensor")},
}
sets["CAN"].update({k.split("_")[0]:d[k] for k in d.files if k.endswith("CAN_only")})
sets["Full"].update({k.split("_")[0]:d[k] for k in d.files if k.endswith("Full_sensor")})
def fold_f1(pred):
    return np.array([f1_score(y[seg==s],pred[seg==s],average='macro') for s in np.unique(seg)])
def mcnemar(a,b):
    ca=a==y; cb=b==y
    n2=int(np.sum(ca&~cb)); n3=int(np.sum(~ca&cb))
    chi=(abs(n2-n3)-1)**2/(n2+n3) if (n2+n3)>0 else 0.0
    return chi,1-stats.chi2.cdf(chi,1)
rows=[]
for setn,md in sets.items():
    base=md["XGB"]; fb=fold_f1(base)
    for name,pred in md.items():
        if name=="XGB": continue
        chi,pm=mcnemar(base,pred); fo=fold_f1(pred)
        try: pt=stats.ttest_rel(fb,fo).pvalue
        except Exception: pt=np.nan
        try: pw=stats.wilcoxon(fb,fo).pvalue
        except Exception: pw=np.nan
        rows.append(dict(FeatureSet=setn,Comparison=f"XGB vs {name}",
                         McNemar_chi2=round(chi,2),McNemar_p=round(pm,4),
                         FoldTtest_p=round(pt,4),Wilcoxon_p=round(pw,4)))
df=pd.DataFrame(rows); df.to_csv(os.path.join(PROJ,"results","significance.csv"),index=False)
print(df.to_string())
