import os,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.chdir(Path(__file__).resolve().parent)
root=Path('.');df=pd.read_csv('analysis/per_run.csv');detail=json.loads(Path('analysis/details.json').read_text());audit=json.loads(Path('analysis/audit.json').read_text())
assert len(df)==28 and df.groupby('suite').size().to_dict()=={'controls':10,'v3_seed0':3,'v4_lowdata':15}, 'Expected all 28 completed runs'
figdir=Path('figures');figdir.mkdir(exist_ok=True)
order=['random','template_init','frozen_templates','random_unitnorm','frozen_random_unitnorm']
low=df[df.suite!='v3_seed0']; v3=df[df.suite=='v3_seed0']
colors=dict(zip(order,['#3465a4','#c65b24','#38804a','#9364ae','#757575']))
def table(headers,rows):
 return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def fmt(s,pct=False,digits=3):
 s=np.asarray(s)*(100 if pct else 1)
 return f'{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}' if len(s)>1 else f'{s.mean():.{digits}f}'
def metric_table(data,keys,headers,pct=False,digits=3):
 return table(['Condition']+headers,[[f'`{r}`']+[fmt(data[data.regime==r][k],pct,digits) for k in keys] for r in order if (data.regime==r).any()])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
for r in order:
 sub=low[low.regime==r]
 axs[0].scatter(sub.align_mean_best,sub.test_acc_sample*100,c=colors[r],label=r,s=35)
 axs[1].scatter(sub.align_mean_best,sub.ood_rot_acc_sample*100,c=colors[r],s=35)
for ax,title in zip(axs,['ID test accuracy','Rotation OOD accuracy']):
 ax.set(xlabel='Kernel alignment: mean_best_score',ylabel='Sample-weighted accuracy (%)',title=title,xlim=(0,1.04));ax.grid(alpha=.15)
axs[0].legend(fontsize=8);fig.suptitle('Low-data runs: each point is one run (dependent adjacent seeds)');fig.tight_layout();fig.savefig(figdir/'accuracy_alignment.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
for r in order:
 suite='v4_lowdata' if r in order[:3] else 'controls'
 curves=np.stack([pd.DataFrame(json.loads(Path(f'runs/{suite}/{r}/seed{s:02d}/train_log.json').read_text())['rows']).val_acc.values for s in range(5)])
 x=np.arange(1,101);m=curves.mean(0)*100;sd=curves.std(0,ddof=1)*100
 ax.plot(x,m,label=r,color=colors[r]);ax.fill_between(x,m-sd,m+sd,color=colors[r],alpha=.12)
ax.set(xlabel='Epoch',ylabel='Repository validation accuracy (%)',title='Low-data learning curves: mean ± sample SD\nOriginal equal-batch weighting; seed dependence applies');ax.legend(fontsize=8);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(figdir/'learning_curves.png');plt.close(fig)
fig,axs=plt.subplots(5,16,figsize=(16,5.5))
for j,r in enumerate(order):
 suite='v4_lowdata' if r in order[:3] else 'controls';W=np.load(f'runs/{suite}/{r}/seed00/conv_W_final.npy')[:,0]
 for i in range(16):
  lim=np.abs(W[i]).max();axs[j,i].imshow(W[i],cmap='RdBu_r',vmin=-lim,vmax=lim);axs[j,i].set_xticks([]);axs[j,i].set_yticks([])
  if j==0:axs[j,i].set_title(str(i),fontsize=8)
 axs[j,0].set_ylabel(r,rotation=0,ha='right',va='center',fontsize=8)
fig.suptitle('Final kernels, seed 0; raw channel order; independent color scale per kernel',fontsize=12);fig.subplots_adjust(left=.20,right=.99,top=.88,bottom=.03,wspace=.05,hspace=.15);fig.savefig(figdir/'kernels_seed0.png');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.5))
for r in order:
 d=[x for x in detail if x['suite']!='v3_seed0' and x['regime']==r]
 for ax,key in zip(axs,['positive_shift','source_prediction']):
  x=np.array([1,2,4,8,16]); vals=np.array([[a['patching'][str(k)]['validation_ranked'][key] for k in x] for a in d]);rnd=np.array([[a['patching'][str(k)]['random_ranked_mean20'][key] for k in x] for a in d]);ax.plot(x,vals.mean(0)*100,color=colors[r],marker='o',label=r);ax.plot(x,rnd.mean(0)*100,color=colors[r],linestyle='--',alpha=.6)
for ax,title in zip(axs,['Positive margin shift','Patched prediction equals source label']):ax.set(xlabel='Number of patched pooled channels',ylabel='Pairs (%)',title=title,xticks=[1,2,4,8,16]);ax.grid(alpha=.15)
axs[0].legend(fontsize=7);fig.suptitle('Validation-ranked channels (solid) vs random rankings (dashed); five-run means');fig.tight_layout();fig.savefig(figdir/'patch_controls.png');plt.close(fig)
fig,axs=plt.subplots(1,4,figsize=(15,4.5),sharey=True)
for ax,split,title in zip(axs,['test','ood_rot','ood_thick','ood_occ'],['ID','Rotation','Thickness','Occlusion']):
 vals=[]
 for r in order:
  cms=[np.array(x['confusion'][split]) for x in detail if x['suite']!='v3_seed0' and x['regime']==r]
  vals.append(np.mean([np.diag(cm)/cm.sum(1) for cm in cms],axis=0)*100)
 vals=np.array(vals);im=ax.imshow(vals,vmin=0,vmax=100,cmap='Blues',aspect='auto')
 ax.set_xticks(range(4),['line','circle','triangle','square'],rotation=45,ha='right');ax.set_yticks(range(5),order);ax.set_title(title)
 for j in range(5):
  for i in range(4):ax.text(i,j,f'{vals[j,i]:.0f}',ha='center',va='center',color='white' if vals[j,i]>65 else 'black',fontsize=8)
fig.suptitle('Per-class recall (%): low-data mean across five dependent seeds');fig.tight_layout();fig.savefig(figdir/'class_recall.png');plt.close(fig)
results='### 4.1 Completion and audit\n\n'
results+=f'All {len(df)} planned training runs completed: 15 v4, 10 added controls, and 3 v3 seed-0 runs. Every checkpoint passed original-ID-accuracy reproduction, ablation-identity, no-op, and full-vector-patching checks. All frozen filters were bitwise unchanged. The 16-entry bank has numerical rank **{audit["rank_tol_1e-6"]}** at singular-value tolerance 10⁻⁶; all four `corner` entries have absolute cosine approximately 1 with an `edge` entry.\n\n'
results+='One added-control launch failed before training because a norm call selected a matrix-norm overload; it was repaired to use an explicit vector norm and successfully rerun. This development failure is retained in `execution.jsonl`; it generated no reported results. The untouched source was independently observed to fail as documented in Section 2.\n\n'
results+='### 4.2 Low-data predictive accuracy\n\nPercentages are sample-weighted **mean ± sample SD** over five paired but dependent seeds. They use final-epoch checkpoints, not validation-best checkpoints.\n\n'
results+=metric_table(low,['test_acc_sample','ood_rot_acc_sample','ood_thick_acc_sample','ood_occ_acc_sample'],['ID (%)','ood_rot (%)','ood_thick (%)','ood_occ (%)'],True,2)
results+='\nThe training-majority classifier scores '+', '.join(f'{label}: {fmt(low[low.regime=="random"][key],True,2)}%' for label,key in [('ID','test_majority_baseline'),('rotation','ood_rot_majority_baseline'),('thickness','ood_thick_majority_baseline'),('occlusion','ood_occ_majority_baseline')])+'. Uniform guessing has expected accuracy 25%; it is not an additional trained model.\n\n'
results+='Original repository equal-batch accuracy, retained for reproduction:\n\n'+metric_table(low,['id_acc','ood_rot_acc','ood_thick_acc','ood_occ_acc'],['id_acc (%)','ood_rot_acc (%)','ood_thick_acc (%)','ood_occ_acc (%)'],True,2)
err=max(abs(low['id_acc']-low['test_acc_sample']).max(),abs(low['ood_rot_acc']-low['ood_rot_acc_sample']).max(),abs(low['ood_thick_acc']-low['ood_thick_acc_sample']).max(),abs(low['ood_occ_acc']-low['ood_occ_acc_sample']).max())
results+=f'\nThe largest original-versus-sample-weighted discrepancy across the low-data final ID/OOD measurements is {err*100:.2f} percentage points.\n\n'
results+='Per-class recall, computed within each run and then averaged, shows where errors concentrate:\n\n![Class recall](figures/class_recall.png)\n\n'
results+='### 4.3 Alignment, drift, and repository probes\n\n'+metric_table(low,['initial_alignment','align_mean_best','drift_cosine','drift_l2','specificity'],['Initial alignment','Final alignment','Drift cosine','Drift L2','Mean specificity'])
results+='\n'+metric_table(low,['patch_single_success','patch_single_shift','patch_topk_success_at_maxk_repo','patch_topk_mean_shift'],['Single success (fraction)','Single mean shift','Best top-k success (fraction)','Top-k mean shift'])
results+='\n“Success” in this table is positive margin movement. The best top-k value is the repository’s `success@maxk` field, maximized over k. It must not be read as classification accuracy or semantic fidelity. Frozen alignment ≈1 is imposed by construction.\n\n![Accuracy and alignment](figures/accuracy_alignment.png)\n\n![Seed-0 kernels](figures/kernels_seed0.png)\n\n'
results+='### 4.4 Learning speed and threshold attainment\n\n'
lrows=[]
for r in order:
 sub=low[low.regime==r];cells=[]
 for k in ['t98','t100']:
  v=sub[k].values;v=v[np.isfinite(v)];cells.append(f'{len(v)}/5; '+(f'median {np.median(v):g}' if len(v) else 'not reached'))
 lrows.append([f'`{r}`',fmt(sub.val_final_repo,True,2),fmt(sub.val_max_repo,True,2),*cells,*[fmt(sub[k],digits=2) for k in ['auc_10','auc_20','auc_50']]])
results+=table(['Condition','Final val (%)','Max val (%)','t98: reached; median epoch','t100: reached; median epoch','AUC10','AUC20','AUC50'],lrows)
results+='\nThreshold medians include only runs that reached the threshold; the accompanying fraction exposes censoring. These are original batch-weighted validation metrics. Max validation is a trajectory summary, not the reported model-selection rule.\n\n![Validation learning curves](figures/learning_curves.png)\n\n'
results+='### 4.5 Controlled comparisons\n\nPaired differences are treatment minus reference, computed within each seed; predictive columns are percentage points. Alignment remains in cosine-score units. Values are mean ± SD of the five paired differences, without significance tests.\n\n'
comparisons=[('template_init','random'),('random_unitnorm','random'),('template_init','random_unitnorm'),('frozen_templates','frozen_random_unitnorm'),('template_init','frozen_templates')]
prows=[]
for a,b in comparisons:
 aa=low[low.regime==a].set_index('seed');bb=low[low.regime==b].set_index('seed');keys=['test_acc_sample','ood_rot_acc_sample','ood_thick_acc_sample','ood_occ_acc_sample','align_mean_best']
 prows.append([f'`{a}` − `{b}`']+[fmt(aa[k]-bb[k],k!='align_mean_best',2 if k!='align_mean_best' else 3) for k in keys])
results+=table(['Treatment − reference','ID pp','Rotation pp','Thickness pp','Occlusion pp','Alignment'],prows)
results+='\n### 4.6 Validation-ranked intervention controls\n\nThe table uses k=8 and the same ID test source/target pairs per seed. Random ranking averages 20 class-specific ranking draws within each run.\n\n'+metric_table(low,['patch_val_rank_k8','patch_random_rank_k8','patch_val_rank_k8_source_prediction','patch_val_rank_k8_source_prediction_both_correct'],['Validation-ranked positive shift (%)','Random-ranked positive shift (%)','Source-label prediction (%)','Source-label prediction, both initially correct (%)'],True,2)
results+='\nThe conditional column uses model-dependent subsets. No-op positive shifts and prediction changes are zero. No-op source-label prediction can be nonzero when the target was already misclassified; per-run values are retained in `analysis/details.json`. At k=16, the source logits are recovered by algebra, irrespective of whether its prediction is correct. These controls demonstrate head-level intervention behavior, not identified causal concepts.\n\n![Patching controls](figures/patch_controls.png)\n\n'
results+='### 4.7 Existing v3 configuration, seed 0\n\nThis is a single-seed reproduction with 8,000 training examples and 10 epochs, not a replicated data-scaling study. Percentages are sample-weighted; there is no seed SD.\n\n'+metric_table(v3,['test_acc_sample','ood_rot_acc_sample','ood_thick_acc_sample','ood_occ_acc_sample'],['ID (%)','ood_rot (%)','ood_thick (%)','ood_occ (%)'],True,2)
results+='\n'+metric_table(v3,['align_mean_best','drift_cosine','specificity','patch_single_success'],['Alignment','Drift cosine','Specificity','Single success (fraction)'])
results+='\nv4 uses 200 optimizer steps (2 batches × 100 epochs); v3 uses 630 (63 × 10). Dataset size, exposure, and update budget therefore differ simultaneously. Compare within each suite; cross-suite differences do not identify an isolated data-size effect. All unabridged metrics, confusion matrices, and intervention curves are retained in the accompanying data.\n'
s=Path('REPORT.template.md').read_text().replace('[Tables and figures inserted from completed runs.]',results)
# The empirical interpretation is supplied separately after manual inspection of generated tables.
for placeholder,file in [('[Results inserted after completion and validation.]','abstract.txt'),('[Interpretation inserted after results validation.]','interpretation.txt'),('[Conclusion inserted after results validation.]','conclusion.txt')]:
 if Path('analysis',file).exists():s=s.replace(placeholder,Path('analysis',file).read_text())
Path('REPORT.md').write_text(s)
print('Report and figures generated for',len(df),'runs')
