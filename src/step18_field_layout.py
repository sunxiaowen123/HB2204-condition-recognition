# -*- coding: utf-8 -*-
"""step18_field_layout.py 场地航迹图（出版级：绝对坐标、匹配地形、比例尺/指北）"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.io import loadmat
sys.path.insert(0, os.path.dirname(__file__)); from config import PROJ
mat=os.path.join(PROJ,"data","plowing_lati_longi.mat")
mm=loadmat(mat, squeeze_me=True)
fig,ax=plt.subplots(figsize=(13,4.6)); colors=plt.cm.tab10(np.linspace(0,1,7))
allx=[]; ally=[]
for i in range(1,8):
    a=np.atleast_1d(mm[f"Channel_28_Data_mat{i}"]).astype(float)
    b=np.atleast_1d(mm[f"Channel_29_Data_mat{i}"]).astype(float)
    ma,mb=np.nanmedian(a),np.nanmedian(b)
    if 40<ma<50: lat,lon=a,b
    elif 40<mb<50: lat,lon=b,a
    else: lat,lon=a,b
    ax.plot(lon,lat,'.',ms=2.5,color=colors[i-1],label=f"segment {i}")
    allx+=list(lon); ally+=list(lat)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)")
ax.set_title("GNSS trajectories of the seven field segments (parallel tillage passes and headland turns)")
ax.set_aspect(1.0/np.cos(np.deg2rad(46.675))); ax.grid(alpha=.3)
ax.legend(markerscale=6, ncol=7, fontsize=8, loc='upper center', bbox_to_anchor=(0.5,-0.16), frameon=False)
# pad
x0d,x1d=np.nanmin(allx),np.nanmax(allx); y0d,y1d=np.nanmin(ally),np.nanmax(ally)
padx=0.06*(x1d-x0d); pady=0.25*(y1d-y0d)
ax.set_xlim(x0d-padx,x1d+padx); ax.set_ylim(y0d-pady,y1d+pady)
dl=100/(111320*np.cos(np.deg2rad(46.675)))
xlim=ax.get_xlim(); ylim=ax.get_ylim()
sx=xlim[0]+0.04*(xlim[1]-xlim[0]); sy=ylim[0]+0.18*(ylim[1]-ylim[0])
ax.plot([sx,sx+dl],[sy,sy],'k-',lw=3); ax.text(sx+dl/2,sy+0.00006,"100 m",ha='center',fontsize=9)
nx=xlim[1]-0.05*(xlim[1]-xlim[0]); ny=ylim[0]+0.62*(ylim[1]-ylim[0])
ax.annotate("N",xy=(nx,ny+1.4*dl),ha='center',fontsize=12,fontweight='bold')
ax.annotate("",xy=(nx,ny+1.7*dl),xytext=(nx,ny+0.5*dl),arrowprops=dict(arrowstyle="->",lw=1.3))
fig.tight_layout(); out=os.path.join(PROJ,"main_text","fig_field_layout.png"); fig.savefig(out,dpi=160,bbox_inches='tight')
print("saved",out)
