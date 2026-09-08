"""Exploratory learning dynamics; added after completed main study."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
rows=[]
for f in sorted((R/'runs').glob('*/*/block*/*/history.json')):
 task,arch,block,condition=f.relative_to(R/'runs').parts[:4]
 h=json.loads(f.read_text());assert len(h)==40
 for x in h:rows.append(dict(task=task,architecture=arch,block=block,condition=condition,**x))
d=pd.DataFrame(rows);assert len(d)==16000
d.to_csv(R/'analysis/learning_curves.csv',index=False)
conditions=['random','random_unitnorm','template_init','template_retention_1','frozen_templates']
settings=[(t,a) for t in ['single_shape','two_concepts'] for a in ['TinyCNN','TwoLayerCNN']]
fig,axes=plt.subplots(2,4,figsize=(18,8),sharex=True)
for j,(t,a) in enumerate(settings):
 for c in conditions:
  s=d[(d.task==t)&(d.architecture==a)&(d.condition==c)]
  for i,m in enumerate(['val_acc','val_ce']):
   g=s.groupby('epoch')[m];mean=g.mean();sd=g.std();x=mean.index.to_numpy();y=mean.to_numpy();e=sd.to_numpy()
   axes[i,j].plot(x,y,label=c);axes[i,j].fill_between(x,y-e,y+e,alpha=.10)
 axes[0,j].set_title(t+'\n'+a);axes[0,j].set_ylim(0,1.03);axes[1,j].set_xlabel('Epoch (4 optimizer steps)')
axes[0,0].set_ylabel('Validation accuracy');axes[1,0].set_ylabel('Validation cross-entropy')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5)
fig.suptitle('Saved learning curves: mean ± SD across 10 independent blocks')
fig.tight_layout(rect=(0,.06,1,.95));fig.savefig(R/'figures/learning_curves.png',dpi=170);plt.close(fig)
summ=[]
for keys,s in d.groupby(['task','architecture','condition','block']):
 s=s.sort_values('epoch');r=dict(zip(['task','architecture','condition','block'],keys))
 r['mean_val_acc_epochs_1_10']=s[s.epoch<=10].val_acc.mean()
 for e in [5,10,20,40]:r['val_acc_epoch_'+str(e)]=float(s[s.epoch==e].val_acc.iloc[0])
 hit=s[s.val_acc>=.95];r['first_epoch_95']=float(hit.epoch.iloc[0]) if len(hit) else np.nan
 r['val_ce_improvement_31_40']=float(s[s.epoch==31].val_ce.iloc[0]-s[s.epoch==40].val_ce.iloc[0])
 summ.append(r)
b=pd.DataFrame(summ);b.to_csv(R/'analysis/learning_dynamics_per_run.csv',index=False)
g=b.groupby(['task','architecture','condition']).mean(numeric_only=True)
g.to_csv(R/'analysis/learning_dynamics_summary.csv')
print(g.loc[(slice(None),slice(None),conditions),['mean_val_acc_epochs_1_10','val_acc_epoch_10','val_acc_epoch_40','val_ce_improvement_31_40']].to_string())
