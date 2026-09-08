from pathlib import Path
import json,itertools
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
R=ROOT/'results/retention_release_001'
h=pd.read_csv(R/'analysis/learning_curves.csv');e=pd.read_csv(R/'analysis/checkpoint_metrics.csv')
assert len(h)==40000 and len(e)==1800
keys=['task','architecture','block','condition']
assert not h.duplicated(keys+['epoch']).any() and not e.duplicated(keys+['epoch']).any()
assert (h.groupby(keys).size()==200).all()
# Independently compare published tables to raw histories/results.
for p in (R/'runs').glob('*/*/block*/*'):
 task,arch,block,c=p.relative_to(R/'runs').parts
 s=h[(h.task==task)&(h.architecture==arch)&(h.block.astype(str).str.replace('block','',regex=False)==str(int(block[5:])))&(h.condition==c)].sort_values('epoch')
 raw=pd.DataFrame(json.loads((p/'history.json').read_text()))
 assert len(s)==200
 for k in raw.select_dtypes('number').columns:assert np.allclose(s[k],raw[k],equal_nan=True), (p,k)
 for f in (p/'evaluations').glob('*/result.json'):
  epoch=int(f.parent.name[6:]);v=json.loads(f.read_text())
  row=e[(e.task==task)&(e.architecture==arch)&(e.block==int(block[5:]))&(e.condition==c)&(e.epoch==epoch)].iloc[0]
  for k,x in v.items():
   if isinstance(x,(int,float)) and not isinstance(x,bool):assert np.isclose(row[k],x,equal_nan=True), (f,k)
conditions=['random','random_unitnorm','template_init','template_retention_1','template_release']
settings=list(itertools.product(['single_shape','two_concepts'],['TinyCNN','TwoLayerCNN']))
colors=dict(zip(conditions,['#3465a4','#e69f00','#009e73','#cc4242','#8255a5']))
rows=[]
for k,s in h.groupby(keys):
 s=s.sort_values('epoch');hits=s[s.val_acc>=.95]
 rows.append(dict(zip(keys,k),early_acc=s[s.epoch<=10].val_acc.mean(),first95=float(hits.epoch.iloc[0]) if len(hits) else np.nan,attained95=len(hits)>0,val_acc40=s[s.epoch==40].val_acc.iloc[0],val_acc200=s.iloc[-1].val_acc,late_ce_slope=np.polyfit(s.tail(20).epoch,s.tail(20).val_ce,1)[0]))
b=pd.DataFrame(rows);b['block']=b.block.astype(str).str.replace('block','',regex=False).astype(int)
f=e[e.epoch==200].merge(b,on=keys,validate='one_to_one');f.to_csv(OUT/'endpoints.csv',index=False)
metrics=['early_acc','first95','id_acc','alignment','semantic_auc','localization_iou','causal_usefulness','cf_accuracy','refit_test_acc','late_ce_slope']
summary=f.groupby(['task','architecture','condition'])[metrics].agg(['mean','std']);summary.to_csv(OUT/'summary.csv')
contrasts=[]
for task,arch in settings:
 for a,base in [('template_init','random_unitnorm'),('template_release','template_retention_1'),('template_release','template_init'),('template_release','random_unitnorm')]:
  sa=f[(f.task==task)&(f.architecture==arch)&(f.condition==a)].set_index('block');sb=f[(f.task==task)&(f.architecture==arch)&(f.condition==base)].set_index('block')
  for m in metrics:
   delta=(sa[m]-sb[m]).dropna();n=len(delta);mean=delta.mean();se=delta.std()/np.sqrt(n);err=t.ppf(.975,n-1)*se
   contrasts.append(dict(task=task,architecture=arch,treatment=a,baseline=base,metric=m,n=n,mean=mean,ci_low=mean-err,ci_high=mean+err))
pd.DataFrame(contrasts).to_csv(OUT/'paired_contrasts.csv',index=False)
for name,df,ms in [('learning',h,['val_acc','val_ce','alignment']),('concepts',e,['semantic_auc','localization_iou','causal_usefulness'])]:
 fig,axes=plt.subplots(3,4,figsize=(17,10),sharex=True)
 for j,(task,arch) in enumerate(settings):
  s=df[(df.task==task)&(df.architecture==arch)]
  for i,m in enumerate(ms):
   for c in conditions:
    g=s[s.condition==c].groupby('epoch')[m];avg=g.mean();sd=g.std();axes[i,j].plot(avg.index,avg,color=colors[c],label=c);axes[i,j].fill_between(avg.index,avg-sd,avg+sd,color=colors[c],alpha=.10)
   axes[i,j].axvline(80,color='grey',ls=':',lw=1);axes[i,j].set_ylabel(m);axes[i,j].grid(alpha=.15)
   if m in ['val_acc','alignment','semantic_auc','localization_iou']:axes[i,j].set_ylim(0,1.02)
  axes[0,j].set_title(task+' / '+arch);axes[2,j].set_xlabel('Epoch')
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5)
 fig.suptitle('Mean ± SD across 10 paired blocks; dotted line: retention release complete')
 fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(OUT/(name+'.png'),dpi=180);plt.close(fig)
# Early curves make first-10-epoch differences visible.
fig,axes=plt.subplots(1,4,figsize=(16,4),sharey=True)
for ax,(task,arch) in zip(axes,settings):
 for c in conditions:
  s=h[(h.task==task)&(h.architecture==arch)&(h.condition==c)&(h.epoch<=20)].groupby('epoch').val_acc.mean();ax.plot(s.index,s,label=c,color=colors[c])
 ax.set_title(task+'\n'+arch);ax.set_xlabel('Epoch');ax.set_ylim(.2,1.02)
axes[0].set_ylabel('Mean validation accuracy');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=5);fig.tight_layout(rect=(0,.1,1,1));fig.savefig(OUT/'early_learning.png',dpi=180);plt.close(fig)
print(f.groupby(['task','architecture','condition'])[metrics].mean().round(4).to_string())
print('Attainment counts:',f.groupby(['task','architecture','condition']).attained95.sum().to_dict())
print('AUDIT PASSED: raw tables match; 200 complete histories, 1800 evaluations')
cs=pd.DataFrame(contrasts);fig,axes=plt.subplots(1,3,figsize=(14,4.5))
for ax,m,label,scale in zip(axes,['id_acc','alignment','causal_usefulness'],['Test accuracy difference (pp)','Alignment difference','Causal usefulness difference'],[100,1,1]):
 for j,(task,arch) in enumerate(settings):
  row=cs[(cs.task==task)&(cs.architecture==arch)&(cs.treatment=='template_release')&(cs.baseline=='template_retention_1')&(cs.metric==m)].iloc[0]
  ax.errorbar(row['mean']*scale,j,xerr=[[scale*(row['mean']-row.ci_low)],[scale*(row.ci_high-row['mean'])]],fmt='o',capsize=4,color='#8255a5')
 ax.axvline(0,color='grey',ls='--');ax.set_yticks(range(4));ax.set_yticklabels([t+' / '+a for t,a in settings] if ax==axes[0] else []);ax.invert_yaxis();ax.set_xlabel(label)
fig.suptitle('Release − constant retention at epoch 200; exploratory marginal 95% paired t intervals')
fig.tight_layout();fig.savefig(OUT/'release_effects.png',dpi=180);plt.close(fig)
# Verify scheduled evaluations and exact pre-release equality of learning measurements.
for task,arch in settings:
 a=h[(h.task==task)&(h.architecture==arch)&(h.condition=='template_release')&(h.epoch<=10)].sort_values(['block','epoch'])
 z=h[(h.task==task)&(h.architecture==arch)&(h.condition=='template_retention_1')&(h.epoch<=10)].sort_values(['block','epoch'])
 for m in ['train_ce','val_ce','val_acc','alignment']:assert np.array_equal(a[m].to_numpy(),z[m].to_numpy())
patches=[json.loads(p.read_text()) for p in (R/'runs').glob('*/*/block*/*/evaluations/*/patching.json')]
assert len(patches)==1800 and all(p['denominator']>=1e-10 for p in patches)
assert all(abs(p['full']['fidelity']-1)<1e-6 and abs(p['noop']['fidelity'])<1e-6 for p in patches)
(OUT/'audit.json').write_text(json.dumps(dict(runs=200,epoch_records=40000,checkpoint_evaluations=1800,raw_tables_match=True,pre_release_histories_identical=True,undefined_fidelity=0,selected_refit_failures=int((~e.refit_converged).sum()),minimum_patch_denominator=min(p['denominator'] for p in patches),scope='Raw JSON/CSV consistency and stored patch identities; does not independently rerun model forwards.'),indent=2)+'\n')
