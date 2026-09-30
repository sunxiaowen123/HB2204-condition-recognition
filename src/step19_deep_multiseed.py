# -*- coding: utf-8 -*-
"""step19_deep_multiseed.py LSTM/TCN 三种子稳健性（LOSO），给 macro-F1 均值±std"""
import os, sys
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(0, os.path.dirname(__file__)); from config import DATA_OUT, OUT, LABELS
torch.set_num_threads(4)
RES=os.path.join(OUT,"results")
d=np.load(os.path.join(DATA_OUT,"window_series.npz")); seq=d["seq"]
meta=pd.read_csv(os.path.join(DATA_OUT,"window_labels.csv")); y=meta.pcgmm.values; g=meta.seg.values
v=seq[:,:,1:2]; seq[:,:,2:5]=np.where(v<0.3,0.0,seq[:,:,2:5])
SETS={"CAN_only":[1,2,3,4],"Full_sensor":[0,1,2,3,4]}
class LSTMNet(nn.Module):
    def __init__(s,c): super().__init__(); s.l=nn.LSTM(c,64,num_layers=2,batch_first=True,dropout=0.2); s.f=nn.Linear(64,5)
    def forward(s,x): x,_=s.l(x); return s.f(x[:,-1,:])
class Chomp1d(nn.Module):
    def __init__(s,k): super().__init__(); s.k=k
    def forward(s,x): return x[:,:,:-s.k] if s.k>0 else x
class TCNBlock(nn.Module):
    def __init__(s,ci,co,k,d,p=.2):
        super().__init__()
        cv=lambda ci,co,d:nn.Sequential(nn.Conv1d(ci,co,k,padding=d*(k-1),dilation=d),Chomp1d(d*(k-1)),nn.ReLU(),nn.Dropout(p))
        s.c1=cv(ci,co,d); s.c2=cv(co,co,d); s.down=nn.Conv1d(ci,co,1) if ci!=co else None; s.act=nn.ReLU()
    def forward(s,x): o=s.c2(s.c1(x)); return s.act(o+(x if s.down is None else s.down(x)))
class TCNNet(nn.Module):
    def __init__(s,c): super().__init__(); s.b=nn.Sequential(TCNBlock(c,32,3,1),TCNBlock(32,32,3,2),TCNBlock(32,32,3,4)); s.f=nn.Linear(32,5)
    def forward(s,x): x=x.transpose(1,2); return s.f(s.b(x).mean(-1))
def train_one(Xtr,ytr,Xte,kind,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu,sd=Xtr.reshape(-1,Xtr.shape[-1]).mean(0),Xtr.reshape(-1,Xtr.shape[-1]).std(0)+1e-6
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    cw=torch.tensor(len(ytr)/(5*np.bincount(ytr,minlength=5)),dtype=torch.float32)
    net={"LSTM":LSTMNet(Xtr.shape[-1]),"TCN":TCNNet(Xtr.shape[-1])}[kind]
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4); lf=nn.CrossEntropyLoss(weight=cw)
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long)
    for ep in range(35):
        net.train(); perm=torch.randperm(len(Xt))
        for i in range(0,len(Xt),64):
            idx=perm[i:i+64]; opt.zero_grad(); lf(net(Xt[idx]),yt[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(torch.tensor(Xte,dtype=torch.float32)).argmax(1).numpy()
segs=sorted(np.unique(g)); seeds=[42,123,2024]; rows=[]
for setn,ch in SETS.items():
    Xall=seq[:,:,ch]
    for kind in ["LSTM","TCN"]:
        agg=[]; folds=[]
        for sd in seeds:
            pred=np.full(len(y),-1)
            for ts in segs: pred[g==ts]=train_one(Xall[g!=ts],y[g!=ts],Xall[g==ts],kind,sd)
            agg.append(f1_score(y,pred,average="macro"))
            folds.append(np.mean([f1_score(y[g==ts],pred[g==ts],average="macro") for ts in segs if len(np.unique(y[g==ts]))>1]))
        rows.append(dict(Model=kind,FeatureSet=setn,AggMacroF1_mean=round(np.mean(agg),3),
                         AggMacroF1_std=round(np.std(agg),3),FoldMacroF1_mean=round(np.mean(folds),3),
                         FoldMacroF1_std=round(np.std(folds),3),seeds=len(seeds)))
        print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES,"deep_multiseed.csv"),index=False)
print("saved deep_multiseed.csv")
