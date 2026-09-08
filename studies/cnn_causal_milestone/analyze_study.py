from __future__ import annotations
import json,hashlib,argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist,ttest_1samp,spearmanr
from study import ROOT,TASKS,ARCHS,CONDITIONS

def estimate(x):
 x=np.asarray(x,dtype=float);x=x[np.isfinite(x)];n=len(x)
 if n==0:return dict(n=0,mean=None,sd=None,lo=None,hi=None,p=None)
 m=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else None
 if n>1:
  half=float(t_dist.ppf(.975,n-1)*sd/np.sqrt(n));p=float(ttest_1samp(x,0).pvalue) if sd>0 else (1. if m==0 else 0.)
 else:half=None;p=None
 return dict(n=n,mean=m,sd=sd,lo=m-half if half is not None else None,hi=m+half if half is not None else None,p=p)
def holm(ps):
 ps=np.array(ps,dtype=float);order=np.argsort(ps);adj=np.empty(len(ps));v=0
 for k,i in enumerate(order):v=max(v,(len(ps)-k)*ps[i]);adj[i]=min(1.,v)
 return adj.tolist()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
 records=[];patches=[];diagnostics=[]
 for p in sorted((ROOT/'runs').glob('*/*/block*/*/result.json')):
  r=json.loads(p.read_text())
  # Exploratory alignment of the independently selected channels themselves.
  from study import bank,normalize
  W=np.load(p.parent/'conv_final.npy')[:,0];A=normalize(W).reshape(16,-1);T=normalize(bank()).reshape(16,-1);best=np.sum(A[:,None,:]*T[None,:,:],axis=2).max(1)
  with np.load(p.parent/'probe_arrays.npz') as z:orders=z['concept_orders']
  r['selected_kernel_alignment']=float(best[orders[:,:4]].mean());r['best_semantic_kernel_alignment']=float(best[orders[:,0]].mean())
  records.append(r)
  pat=json.loads((p.parent/'patching.json').read_text());pat.update({k:r[k] for k in ['task','architecture','block','condition']});patches.append(pat)
  r['input_effect_mean_square']=pat['denominator']/pat['pair_count'];r['counterfactual_agreement']=pat['selected']['4']['agreement']
  heads=json.loads((p.parent/'head_refit.json').read_text());fit=heads['selected_fit'];diagnostics.append({k:r[k] for k in ['task','architecture','block','condition']}|{k:fit[k] for k in ['alpha','success','iterations','gradient_max','train_acc','val_ce','val_acc','test_acc']}|{'gain_test':fit['test_acc']-r['id_acc']})
 if not a.partial:assert len(records)==400,f'{len(records)}/400 runs complete'
 out=ROOT/('analysis_partial' if a.partial else 'analysis');out.mkdir(exist_ok=True)
 df=pd.DataFrame(records);df.to_csv(out/'per_run.csv',index=False);pd.DataFrame(diagnostics).to_csv(out/'head_diagnostics.csv',index=False)
 (out/'patching_all.json').write_text(json.dumps(patches,indent=2))
 metrics=['id_acc','alignment','semantic_auc','localization_iou','fidelity_selected','fidelity_random','causal_usefulness','cf_accuracy','cf_accuracy_random','refit_test_acc','nuisance_acc','nuisance_prediction_stability','nuisance_probability_l2','drift_l2','selected_kernel_alignment','best_semantic_kernel_alignment','input_effect_mean_square','counterfactual_agreement']
 summaries=[]
 for (task,arch,condition),sub in df.groupby(['task','architecture','condition']):
  for metric in metrics:
   e=estimate(sub[metric]);e.pop('p');summaries.append(dict(task=task,architecture=arch,condition=condition,metric=metric,**e))
 pd.DataFrame(summaries).to_csv(out/'summary.csv',index=False)
 primary=[];pairs=[]
 comparisons=[('template_retention_1','template_init'),('template_retention_0.1','template_init'),('template_init','random_unitnorm'),('template_init','spectrum_init'),('template_retention_1','spectrum_retention_1'),('frozen_templates','frozen_random_unitnorm'),('frozen_templates','frozen_spectrum')]
 for task in TASKS:
  for arch in ARCHS:
   sub=df[(df.task==task)&(df.architecture==arch)]
   for cond,base in comparisons:
    s=sub[sub.condition==cond].set_index('block');b=sub[sub.condition==base].set_index('block');ix=s.index.intersection(b.index)
    if len(ix)==0:continue
    for metric in metrics:
     delta=s.loc[ix,metric]-b.loc[ix,metric];e=estimate(delta)
     row=dict(task=task,architecture=arch,treatment=cond,reference=base,metric=metric,**e)
     pairs.append(row)
     if cond=='template_retention_1' and base=='template_init' and metric=='causal_usefulness':
      row=row.copy();row['alignment_delta']=float((s.loc[ix,'alignment']-b.loc[ix,'alignment']).mean());row['accuracy_delta']=float((s.loc[ix,'id_acc']-b.loc[ix,'id_acc']).mean());row['semantic_auc_delta']=float((s.loc[ix,'semantic_auc']-b.loc[ix,'semantic_auc']).mean());primary.append(row)
 if len(primary)==4 and all(x['p'] is not None for x in primary):
  for r,p in zip(primary,holm([x['p'] for x in primary])):
   r['p_holm']=p;r['strong_alignment_manipulation']=r['alignment_delta']>=.1;r['accuracy_tradeoff_flag']=r['accuracy_delta']<-.05
   if r['alignment_delta']<=0:r['decision']='alignment manipulation not demonstrated'
   elif p<.05 and r['mean']<0:r['decision']='negative causal-usefulness effect'
   elif p<.05 and r['mean']>0:r['decision']='positive effect with accuracy trade-off' if r['accuracy_tradeoff_flag'] else 'positive effect without >5 pp accuracy loss'
   else:r['decision']='inconclusive causal-usefulness effect'
 pd.DataFrame(primary).to_csv(out/'primary_contrasts.csv',index=False);pd.DataFrame(pairs).to_csv(out/'paired_contrasts.csv',index=False)
 correlations=[]
 for (task,arch),sub in df.groupby(['task','architecture']):
  for target in ['semantic_auc','causal_usefulness','id_acc']:
   r=spearmanr(sub.alignment,sub[target],nan_policy='omit');correlations.append(dict(task=task,architecture=arch,target=target,spearman_rho=float(r.statistic),note='Descriptive, pooled treatments within setting; no independent-observation p-value reported.'))
 pd.DataFrame(correlations).to_csv(out/'descriptive_correlations.csv',index=False)
 print(json.dumps({'runs':len(df),'primary':primary},indent=2))
if __name__=='__main__':main()
