import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from study import *
def main():
 (ROOT/'figures').mkdir(exist_ok=True)
 fig,ax=plt.subplots(3,8,figsize=(12,5.5));T=bank();S=spectrum_bank(T,0)
 for i in range(8):
  for row,W,label in [(0,T[i],NAMES[i]),(1,T[8+i],NAMES[8+i]),(2,S[8+i],'spectrum '+NAMES[8+i])]:
   v=np.abs(W).max();ax[row,i].imshow(W,cmap='RdBu_r',vmin=-v,vmax=v);ax[row,i].set_title(label,fontsize=7)
 for a in ax.flat:a.axis('off')
 fig.suptitle('Corrected templates and spectrum controls\nSymmetric individual color scales; zero is white');fig.tight_layout(rect=[0,0,1,.9]);fig.savefig(ROOT/'figures/bank_audit.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(4,4,figsize=(8,8))
 for j,task in enumerate(TASKS):
  d=data_for(task,0)['test']
  for s in range(4):
   ax[j*2,s].imshow(d['x'][s,0],cmap='gray',vmin=0,vmax=1);ax[j*2,s].set_title(f'{task}: state {s}',fontsize=8)
   ax[j*2+1,s].imshow(d['x'][s,0],cmap='gray',vmin=0,vmax=1);ms=d['masks'][s].any(0)
   if ms.any():ax[j*2+1,s].contour(ms,levels=[.5],colors=['#e24a33'],linewidths=.8)
   ax[j*2+1,s].set_title('concept-positive foreground mask',fontsize=7)
 for a in ax.flat:a.axis('off')
 fig.suptitle('Matched identity changes; nuisance factors held fixed');fig.tight_layout();fig.savefig(ROOT/'figures/matched_inputs.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
