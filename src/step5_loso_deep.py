# -*- coding: utf-8 -*-
"""15_loso_deep.py  LSTM / TCN 的 LOSO-7折（60点@20Hz窗口序列）
序列通道: [Fx,v,Slip_LF,Slip_LR,Slip_RR]; CAN_only用后4通道, Full_sensor用全部5通道
每折内通道标准化、低速步滑转置0、类别加权; CPU训练,固定种子。
"""
import os, sys, json
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_OUT, OUT, LABELS, SEED
torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED)
RES=os.path.join(OUT,"results")

d=np.load(os.path.join(DATA_OUT,"window_series.npz"))
seq=d['seq']; meta=pd.read_csv(os.path.join(DATA_OUT,"window_labels.csv"))
y=meta.pcgmm.values; g=meta.seg.values
# 低速步(v<0.3)滑转物理无定义->0
v=seq[:,:,1:2]; seq[:,:,2:5]=np.where(v<0.3,0.0,seq[:,:,2:5])
SETS={'CAN_only':[1,2,3,4],'Full_sensor':[0,1,2,3,4]}

class LSTMNet(nn.Module):
    def __init__(s,c,h=64):
        super().__init__(); s.l=nn.LSTM(c,h,num_layers=2,batch_first=True,dropout=0.2); s.f=nn.Linear(h,5)
    def forward(s,x): x,_=s.l(x); return s.f(x[:,-1,:])
class Chomp1d(nn.Module):
    def __init__(s,k): super().__init__(); s.k=k
    def forward(s,x): return x[:,:,:-s.k] if s.k>0 else x
class TCNBlock(nn.Module):
    def __init__(s,ci,co,k,d,p=0.2):
        super().__init__()
        conv=lambda:nn.Sequential(nn.Conv1d(ci,co,k,padding=d*(k-1),dilation=d),Chomp1d(d*(k-1)),
                                  nn.ReLU(),nn.Dropout(p))
        s.c1=conv(); s.c2=nn.Sequential(nn.Conv1d(co,co,k,padding=d*(k-1),dilation=d),
                                        Chomp1d(d*(k-1)),nn.ReLU(),nn.Dropout(p))
        s.down=nn.Conv1d(ci,co,1) if ci!=co else None; s.act=nn.ReLU()
    def forward(s,x): o=s.c2(s.c1(x)); return s.act(o+(x if s.down is None else s.down(x)))
class TCNNet(nn.Module):
    def __init__(s,c,ch=32):
        super().__init__()
        s.b=nn.Sequential(TCNBlock(c,ch,3,1),TCNBlock(ch,ch,3,2),TCNBlock(ch,ch,3,4))
        s.f=nn.Linear(ch,5)
    def forward(s,x):
        x=x.transpose(1,2); o=s.b(x); o=o.mean(-1); return s.f(o)

def train_one(Xtr,ytr,Xte,kind,epochs=35):
    mu,sd=Xtr.reshape(-1,Xtr.shape[-1]).mean(0),Xtr.reshape(-1,Xtr.shape[-1]).std(0)+1e-6
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    cw=torch.tensor(len(ytr)/(5*np.bincount(ytr,minlength=5)),dtype=torch.float32)
    net={'LSTM':LSTMNet(Xtr.shape[-1]),'TCN':TCNNet(Xtr.shape[-1])}[kind]
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.CrossEntropyLoss(weight=cw)
    Xt=torch.tensor(Xtr,dtype=torch.float32); yt=torch.tensor(ytr,dtype=torch.long)
    n=len(Xt); bs=64
    for ep in range(epochs):
        net.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            idx=perm[i:i+bs]; opt.zero_grad()
            loss=lossf(net(Xt[idx]),yt[idx]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): pred=net(torch.tensor(Xte,dtype=torch.float32)).argmax(1).numpy()
    return pred

segs=sorted(np.unique(g)); rows=[]; oof={}
for setname,ch in SETS.items():
    Xall=seq[:,:,ch]
    for kind in ['LSTM','TCN']:
        pred_all=np.full(len(y),-1); fold=[]
        for ts in segs:
            tr,te=g!=ts,g==ts
            pr=train_one(Xall[tr],y[tr],Xall[te],kind); pred_all[te]=pr
            if len(np.unique(y[te]))>1: fold.append(f1_score(y[te],pr,average='macro'))
        acc=accuracy_score(y,pred_all); mf1=f1_score(y,pred_all,average='macro')
        bacc=balanced_accuracy_score(y,pred_all); pc=f1_score(y,pred_all,average=None,labels=range(5))
        oof[f"{kind}_{setname}"]=pred_all
        rows.append(dict(Model=kind,FeatureSet=setname,Acc=acc,MacroF1=mf1,BalAcc=bacc,
                         FoldF1_mean=np.mean(fold),FoldF1_std=np.std(fold),
                         **{f"F1_{LABELS[i]}":pc[i] for i in range(5)}))
        print(f"{kind:4s} {setname:11s} Acc={acc:.3f} macroF1={mf1:.3f} balAcc={bacc:.3f} "
              f"Trans_F1={pc[2]:.3f} foldF1={np.mean(fold):.3f}±{np.std(fold):.3f}",flush=True)
R=pd.DataFrame(rows); R.to_csv(os.path.join(RES,"deep_results.csv"),index=False)
np.savez(os.path.join(RES,"oof_deep.npz"),**oof,y=y,seg=g)
print("saved deep_results.csv / oof_deep.npz")
