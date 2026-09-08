import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import ROOT,TASKS,ARCHS,CONDITIONS
ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args()
adir=ROOT/('analysis_partial' if args.partial else 'analysis');df=pd.read_csv(adir/'per_run.csv');primary=pd.read_csv(adir/'primary_contrasts.csv');diag=pd.read_csv(adir/'head_diagnostics.csv');patch=json.loads((adir/'patching_all.json').read_text())
if not args.partial:assert len(df)==400
figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
from make_design_figures import main as make_design_figures
make_design_figures()
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
palette={'random':'#2365a1','random_unitnorm':'#2365a1','frozen_random_unitnorm':'#2365a1','template_init':'#c95420','template_retention_0.1':'#c95420','template_retention_1':'#c95420','frozen_templates':'#c95420','spectrum_init':'#7857a6','spectrum_retention_1':'#7857a6','frozen_spectrum':'#7857a6'}
markers={'random':'x','random_unitnorm':'o','frozen_random_unitnorm':'s','template_init':'o','template_retention_0.1':'^','template_retention_1':'D','frozen_templates':'s','spectrum_init':'o','spectrum_retention_1':'D','frozen_spectrum':'s'}
settings=[(t,a) for t in TASKS for a in ARCHS]
def tab(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def fmt(x,pct=False):
 x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]*(100 if pct else 1)
 return 'undefined' if len(x)==0 else (f'{x.mean():.2f} ± {x.std(ddof=1):.2f}' if pct else f'{x.mean():.3f} ± {x.std(ddof=1):.3f}')
def ci(row):return f"{row['mean']:+.3f} [{row['lo']:+.3f}, {row['hi']:+.3f}]"
for metric,label,name in [('causal_usefulness','Causal usefulness U','alignment_usefulness'),('semantic_auc','Concept selectivity (AUROC)','alignment_semantics')]:
 fig,axs=plt.subplots(2,2,figsize=(11,8))
 for ax,(task,arch) in zip(axs.flat,settings):
  sub=df[(df.task==task)&(df.architecture==arch)]
  for cond in CONDITIONS:
   d=sub[sub.condition==cond]
   if len(d)==0:continue
   ax.errorbar(d.alignment.mean(),d[metric].mean(),xerr=d.alignment.std(ddof=1),yerr=d[metric].std(ddof=1),fmt=markers[cond],color=palette[cond],markersize=5,capsize=2,alpha=.8,label=cond)
  ts=sub[sub.condition.isin(['template_init','template_retention_0.1','template_retention_1'])].groupby('condition')[['alignment',metric]].mean().reindex(['template_init','template_retention_0.1','template_retention_1'])
  ax.plot(ts.alignment,ts[metric],color='#c95420',alpha=.45,linewidth=1)
  ax.set(title=f'{task} / {arch}',xlabel='Corrected-bank alignment',ylabel=label,xlim=(0,1.04));ax.axhline(0,color='gray',linewidth=.6,alpha=.5);ax.grid(alpha=.13)
 bounds=df.groupby(['task','architecture','condition'])[metric].agg(['mean','std'])
 if metric=='causal_usefulness':
  yl=(min(-.05,float((bounds['mean']-bounds['std']).min())-.03),max(.05,float((bounds['mean']+bounds['std']).max())+.03))
 else:yl=(.5,1.02)
 for ax in axs.flat:ax.set_ylim(*yl)
 handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8,bbox_to_anchor=(.5,-.03));fig.suptitle('Condition means ± block SD; orange line follows λ = 0 → 0.1 → 1');fig.tight_layout(rect=[0,.15,1,.95]);fig.savefig(figdir/f'{name}.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.5))
for i,r in primary.iterrows():
 ax.errorbar(r['mean'],i,xerr=np.array([[r['mean']-r['lo']],[r['hi']-r['mean']]]),fmt='o',color='#c95420',capsize=4)
 ax.annotate(f"Holm p={r.get('p_holm',float('nan')):.3g}",(r['hi'],i),xytext=(8,0),textcoords='offset points',va='center',fontsize=8)
ax.set_yticks(range(len(primary)),[f'{r.task} / {r.architecture}' for r in primary.itertuples()]);ax.axvline(0,color='gray',linestyle='--');ax.set(xlabel='Paired change in U: template_retention_1 − template_init',title='Four prospective contrasts: mean and marginal 95% t interval');ax.invert_yaxis();ax.margins(x=.4);fig.tight_layout();fig.savefig(figdir/'primary_effects.png',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11,7.5))
frozen=['frozen_random_unitnorm','frozen_spectrum','frozen_templates']
for ax,(task,arch) in zip(axs.flat,settings):
 sub=df[(df.task==task)&(df.architecture==arch)]
 for i,c in enumerate(frozen):
  d=sub[sub.condition==c];ax.barh(i-.17,d.id_acc.mean()*100,height=.32,color=palette[c],alpha=.45,xerr=d.id_acc.std()*100,capsize=2);ax.barh(i+.17,d.refit_test_acc.mean()*100,height=.32,color=palette[c],xerr=d.refit_test_acc.std()*100,capsize=2)
 ax.set_yticks(range(3),['random unit norm','spectrum','templates']);ax.set(xlim=(0,105),xlabel='Test accuracy (%)',title=f'{task} / {arch}');ax.grid(axis='x',alpha=.15)
fig.suptitle('Frozen first-layer conditions: trained head (light) vs validation-selected refit (dark)\nMean ± block SD; second convolution remains trainable in TwoLayerCNN');fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(figdir/'head_optimization.png');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(10,7))
for ax,(task,arch) in zip(axs.flat,settings):
 for cond in ['random_unitnorm','template_init','template_retention_1','spectrum_retention_1']:
  p=[r for r in patch if r['task']==task and r['architecture']==arch and r['condition']==cond]
  if not p:continue
  vals=np.array([[r['selected'][str(k)]['fidelity'] for k in [1,4,8]] for r in p]);ax.plot([1,4,8],vals.mean(0),marker=markers[cond],color=palette[cond],linestyle='--' if 'retention' in cond else '-',label=cond)
 ax.set(title=f'{task} / {arch}',xlabel='Selected channels',ylabel='Effect fidelity F',xticks=[1,4,8]);ax.grid(alpha=.15)
handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8);fig.suptitle('Selected-channel fidelity; k=4 is primary, k=1 and k=8 are exploratory');fig.tight_layout(rect=[0,.1,1,.95]);fig.savefig(figdir/'patch_size.png');plt.close(fig)
results='### 4.1 Completion and validity\n\n'
results+=f'{len(df)} runs are included. '+('All 400 planned runs completed; every condition has ten blocks in each setting. ' if len(df)==400 else 'This is an incomplete preview. ')
results+='The final audit independently verifies sample-weighted checkpoint accuracy, alignment, full/no-op patch identities, classifier-refit predictions, exact data pairing, and source/protocol integrity. The first-pass report and uploaded-source working files remain unchanged.\n\n'
results+='### 4.2 Predictive accuracy, alignment, semantics and causal usefulness\n\nValues are mean ± sample SD across independent blocks. Accuracy is percent; alignment, AUROC, IoU and U are dimensionless. U is selected k=4 fidelity minus the eight-random-ranking mean. All labels and nomenclature match this milestone’s code.\n\n'
for task,arch in settings:
 sub=df[(df.task==task)&(df.architecture==arch)]
 results+=f'**{task} / {arch}**\n\n'
 results+=tab(['Condition','ID accuracy (%)','Alignment','Concept AUROC','Mask IoU','U'],[[f'`{c}`']+[fmt(sub[sub.condition==c][k],k=='id_acc') for k in ['id_acc','alignment','semantic_auc','localization_iou','causal_usefulness']] for c in CONDITIONS if (sub.condition==c).any()])+'\n'
results+='![Alignment versus causal usefulness](figures/alignment_usefulness.png)\n\n![Alignment versus semantic selectivity](figures/alignment_semantics.png)\n\n'
results+='### 4.3 Prospective retention contrasts\n\nTreatment is `template_retention_1`; reference is `template_init`. Δaccuracy is in percentage points. ΔU intervals are marginal 95% intervals; p-values are Holm-adjusted across these four tests.\n\n'
results+=tab(['Task / architecture','Δalignment','Δaccuracy pp','ΔU [95% CI]','Holm p','Operational decision'],[[f"{r['task']} / {r['architecture']}",f"{r['alignment_delta']:+.3f}",f"{r['accuracy_delta']*100:+.2f}",ci(r),f"{r.get('p_holm',float('nan')):.4g}",r.get('decision','pending')] for r in primary.to_dict('records')])
results+='\nPositive or negative decisions apply to the predeclared U measure and setting. A negative effect does not prove every aligned channel is useless. A positive effect does not establish a generally identified causal representation.\n\n![Primary paired effects](figures/primary_effects.png)\n\n'
results+='### 4.4 Actual counterfactual prediction and fidelity\n\nThis table focuses on the primary treatment pair and its spectrum counterpart. “Full CF accuracy” equals model ID accuracy because incoming counterfactual states are balanced. High fidelity alone does not imply a correct counterfactual prediction.\n\n'
rows=[]
for task,arch in settings:
 for c in ['template_init','template_retention_1','spectrum_retention_1']:
  d=df[(df.task==task)&(df.architecture==arch)&(df.condition==c)]
  rows.append([f'{task} / {arch}',c,fmt(d.fidelity_selected),fmt(d.fidelity_random),fmt(d.cf_accuracy,True),fmt(d.cf_accuracy_random,True),fmt(d.id_acc,True),fmt(d.input_effect_mean_square)])
results+=tab(['Setting','Condition','Selected F','Random F','Selected CF accuracy (%)','Random CF accuracy (%)','Full CF accuracy (%)','Mean squared input effect'],rows)
results+='\nAll per-run agreement, correct-pair subsets, input-effect denominators and k-dependent results are preserved in `analysis/patching_all.json`.\n\n![Patch size dependence](figures/patch_size.png)\n\n'
results+='### 4.5 Frozen-model optimization diagnostic\n\nThe alternative head is selected by validation loss; these results do not replace the main model’s causal metrics. Gains are paired within each block.\n\n'
rows=[]
for task,arch in settings:
 for c in frozen:
  d=df[(df.task==task)&(df.architecture==arch)&(df.condition==c)];h=diag[(diag.task==task)&(diag.architecture==arch)&(diag.condition==c)]
  rows.append([f'{task} / {arch}',c,fmt(d.id_acc,True),fmt(d.refit_test_acc,True),fmt(d.refit_test_acc-d.id_acc,True),f'{int(h.success.sum())}/{len(h)}'])
results+=tab(['Setting','Frozen condition','Original head (%)','Refitted head (%)','Gain pp','Selected solver converged'],rows)
results+=f'\nAcross all conditions, {int(diag.success.sum())}/{len(diag)} selected fits report successful solver termination. Individual gradient norms and all candidates are retained; a successful termination flag is not an information-theoretic capacity certificate.\n\n![Frozen-head diagnostic](figures/head_optimization.png)\n\n'
results+='**Exploratory classifier sensitivity of patching.** This supplement was motivated by partial outcomes and changes only the head to its already validation-selected refit. Features and channel rankings remain fixed; these values do not replace the primary tests. Full detail for 200 evaluations is in `analysis/head_causal_diagnostic.csv`.\n\n'
hp=ROOT/'analysis/head_causal_diagnostic.csv'
if hp.exists():
 hd=pd.read_csv(hp)
 if not args.partial:assert len(hd)==200
 rows=[]
 for task,arch in settings:
  for c in ['template_init','template_retention_1','frozen_templates']:
   d=hd[(hd.task==task)&(hd.architecture==arch)&(hd.condition==c)]
   rows.append([f'{task} / {arch}',c,fmt(d.original_U),fmt(d.refit_U),fmt(d.delta_U),fmt(d.refit_cf_accuracy,True)])
 results+=tab(['Setting','Condition','Original U','Refit U','ΔU (exploratory)','Refit selected CF accuracy (%)'],rows)+'\n'
elif not args.partial:raise RuntimeError('Missing exploratory head diagnostic')
results+='Alignment of the validation-selected channels themselves is additionally recorded in the per-run table (`selected_kernel_alignment`, `best_semantic_kernel_alignment`), without changing their selection. These are exploratory checks on using a whole-layer alignment average.\n\n'
results+='### 4.6 Matched nuisance stability\n\nNuisance changes alter translation and pixel noise jointly while preserving concepts. This is a separate label-stability diagnostic.\n\n'
rows=[]
for task,arch in settings:
 for c in ['random_unitnorm','template_init','template_retention_1','frozen_templates']:
  d=df[(df.task==task)&(df.architecture==arch)&(df.condition==c)];rows.append([f'{task} / {arch}',c,fmt(d.nuisance_acc,True),fmt(d.nuisance_prediction_stability,True),fmt(d.nuisance_probability_l2)])
results+=tab(['Setting','Condition','Nuisance-image accuracy (%)','Prediction stability (%)','Mean squared probability change'],rows)
results+='\nExploratory first-layer concept-versus-nuisance response magnitudes are saved in `analysis/nuisance_features.csv`. These features are not used to choose the primary channels or change training.\n'
s=(ROOT/'REPORT.template.md').read_text().replace('{{RESULTS}}',results)
for key,file in [('ABSTRACT','abstract.txt'),('INTERPRETATION','interpretation.txt'),('NEXT','next.txt'),('CONCLUSION','conclusion.txt')]:
 p=ROOT/'analysis'/file
 if p.exists():s=s.replace('{{'+key+'}}',p.read_text())
(ROOT/('REPORT.preview.md' if args.partial else 'REPORT.md')).write_text(s)
print('Report generated for',len(df),'runs')
