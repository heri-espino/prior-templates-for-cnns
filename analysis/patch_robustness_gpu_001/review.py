"""Review uploaded robustness outcomes; no training or automatic confirmatory gate.

This script intentionally reuses the already-uploaded Stage C checkpoint evaluations.
It now decomposes U = selected fidelity - random fidelity so intervention-budget
sensitivity can be attributed to its observed components rather than to U alone.
"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
R=ROOT/'results/patch_robustness_gpu_001'

rows=[]
legacy=[]
files=sorted((R/'runs').glob('*/*/block*/*/*.json'))
assert len(files)==80
for p in files:
 v=json.loads(p.read_text())
 identity={k:v[k] for k in ['task','architecture','block','condition','epoch']}
 assert len(v['rows'])==12 and v['denominator']>=1e-10
 assert v['full_patch_max_error']<2e-5 and v['noop_max_error']<2e-5
 assert all(v['validation_patch_defined'])
 rows.extend(dict(**identity,**x) for x in v['rows'])
 legacy.append(v['legacy_difference'])

df=pd.DataFrame(rows)
assert len(df)==960
assert not df.duplicated(['task','architecture','block','condition','epoch','method','k']).any()
identity_error=(df['causal_usefulness']-(df['fidelity']-df['random_fidelity'])).abs().max()
assert identity_error<1e-12
# Preserve the complete per-checkpoint decomposition for direct inspection.
df.to_csv(OUT/'method_size_rows.csv',index=False)

METRICS=['fidelity','random_fidelity','causal_usefulness','cf_accuracy','random_cf_accuracy']
contrasts=[]
for key,s in df.groupby(['task','architecture','method','k']):
 for metric in METRICS:
  a=s[s.condition=='template_release'].set_index('block')[metric]
  b=s[s.condition=='template_retention_1'].set_index('block')[metric]
  delta=(a-b).dropna()
  assert len(delta)==10
  mean=delta.mean()
  err=t.ppf(.975,9)*delta.std()/np.sqrt(10)
  contrasts.append(dict(zip(['task','architecture','method','k'],key),metric=metric,n=10,mean=mean,ci_low=mean-err,ci_high=mean+err))
c=pd.DataFrame(contrasts)
c.to_csv(OUT/'paired_contrasts.csv',index=False)

# Original reader-facing sensitivity plot: relative U and absolute CF accuracy.
fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
for j,arch in enumerate(['TinyCNN','TwoLayerCNN']):
 for i,metric in enumerate(['causal_usefulness','cf_accuracy']):
  ax=axes[i,j]
  for offset,method in zip([-.1,0,.1],['contrast','auroc','validation_patch']):
   s=c[(c.task=='two_concepts')&(c.architecture==arch)&(c.metric==metric)&(c.method==method)].sort_values('k')
   ax.errorbar(np.arange(4)+offset,s['mean'],yerr=[s['mean']-s.ci_low,s.ci_high-s['mean']],marker='o',capsize=3,label=method)
  ax.axhline(0,color='grey',ls='--')
  ax.set_xticks(range(4));ax.set_xticklabels([1,2,4,8]);ax.set_xlabel('Patched channels')
  ax.set_title(arch);ax.set_ylabel('Release - constant: '+metric)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3)
fig.suptitle('two_concepts: exploratory marginal 95% paired intervals (10 blocks)')
fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(OUT/'measurement_sensitivity.png',dpi=170);plt.close(fig)

# New decomposition: selected fidelity, same-size random fidelity, and their difference U.
fig,axes=plt.subplots(3,2,figsize=(11,11),sharex=True)
for j,arch in enumerate(['TinyCNN','TwoLayerCNN']):
 for i,metric in enumerate(['fidelity','random_fidelity','causal_usefulness']):
  ax=axes[i,j]
  methods=['contrast','auroc','validation_patch'] if metric!='random_fidelity' else ['contrast']
  offsets=[-.1,0,.1] if metric!='random_fidelity' else [0]
  labels=methods
  for offset,method,label in zip(offsets,methods,labels):
   s=c[(c.task=='two_concepts')&(c.architecture==arch)&(c.metric==metric)&(c.method==method)].sort_values('k')
   ax.errorbar(np.arange(4)+offset,s['mean'],yerr=[s['mean']-s.ci_low,s.ci_high-s['mean']],marker='o',capsize=3,label=label)
  ax.axhline(0,color='grey',ls='--')
  ax.set_xticks(range(4));ax.set_xticklabels([1,2,4,8]);ax.set_xlabel('Patched channels')
  ax.set_title(arch if i==0 else '')
  ax.set_ylabel({'fidelity':'Δ selected fidelity','random_fidelity':'Δ random fidelity','causal_usefulness':'Δ U'}[metric])
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3)
fig.suptitle('two_concepts: decomposition of release - constant patching contrasts')
fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(OUT/'fidelity_decomposition.png',dpi=170);plt.close(fig)

# Compact decomposition table for the compositional task.
def cell(row):
 return f"{row['mean']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"

table=[]
for arch in ['TinyCNN','TwoLayerCNN']:
 for method in ['contrast','auroc','validation_patch']:
  for k in [1,2,4,8]:
   rec={'architecture':arch,'method':method,'k':k}
   for metric in ['fidelity','random_fidelity','causal_usefulness','cf_accuracy']:
    row=c[(c.task=='two_concepts')&(c.architecture==arch)&(c.method==method)&(c.k==k)&(c.metric==metric)].iloc[0]
    rec[metric]=cell(row)
   table.append(rec)
pd.DataFrame(table).to_csv(OUT/'two_concepts_component_contrasts.csv',index=False)

md=['| Architecture | Ranking | k | Δ selected F | Δ random F | ΔU | Δ CF accuracy |',
    '|---|---|---:|---:|---:|---:|---:|']
for r in table:
 md.append(f"| {r['architecture']} | {r['method']} | {r['k']} | {r['fidelity']} | {r['random_fidelity']} | {r['causal_usefulness']} | {r['cf_accuracy']} |")
component_table='\n'.join(md)

max_u=max(abs(v['causal_usefulness']) for v in legacy)
max_acc=max(abs(v['test_acc']) for v in legacy)
(OUT/'audit.json').write_text(json.dumps(dict(
 checkpoints=80,
 method_size_rows=960,
 paired_blocks=10,
 raw_checks_passed=True,
 usefulness_identity_max_abs_error=float(identity_error),
 maximum_legacy_U_difference=max_u,
 maximum_legacy_accuracy_difference=max_acc,
 scope='Raw stored results and patch identities; not independent rerun of GPU forwards.'
),indent=2)+'\n')

(OUT/'REPORT.md').write_text(f'''# Does the causal conclusion survive measurement changes?

All 80 GPU evaluations are present. Raw JSON yielded 960 method/size records. Maximum original-k4 U difference versus stored CPU results: {max_u:.3g}; test accuracy difference: {max_acc:.3g}. All stored full/no-op checks and validation-selection denominators passed. The stored identity `U = fidelity - random_fidelity` is reproduced with maximum absolute error {identity_error:.3g}.

![Measurement sensitivity](measurement_sensitivity.png)

## Finding

The result survives **particular** changes, not all changes. On two_concepts / TinyCNN, all three selection methods yield positive release-minus-constant U at k4 and k8, but negative differences at k1 and k2. Thus the apparent advantage is patch-budget dependent and not merely an artifact of the original contrast ranking.

For TwoLayerCNN, contrast and AUROC show negative differences at k4; validation-patch ranking gives a negative mean with an interval spanning zero. At k8, validation-patch ranking gives a positive difference. A universal claim that release improves shallow causal usefulness and harms deeper causal usefulness is therefore too broad.

These are reused checkpoints/test data, post hoc analyses, and marginal intervals without multiplicity adjustment. A sign change is descriptive evidence of measurement dependence; it does not alone establish the mechanism. Singleton-based validation-patch ranking does not optimize joint subsets.

## Decomposing U

`causal_usefulness` is a difference of two size-dependent quantities:

`U(k) = selected_fidelity(k) - random_fidelity(k)`.

The figure and table below expose both components. This is necessary because a treatment difference in U can change sign through selected fidelity, the random baseline, or both. The decomposition is an analysis of the same Stage C outputs, not independent evidence.

![Fidelity decomposition](fidelity_decomposition.png)

{component_table}

Interpret the component table before making any mechanistic statement about concentration or distribution of causal information. In particular, a reversal in ΔU is not sufficient evidence for redistribution if the selected-fidelity contrast does not show the corresponding pattern.

## Decision for the conditional replication

**Defer confirmation of any broad architecture claim until the component decomposition and an intervention-magnitude-matched control have been reviewed.** The original condition—robustness across measurement choices—is not established. This is a scientific assessment, not a predeclared statistical pass/fail threshold.

A narrower next question is whether release changes the treatment difference across channel-set sizes in selected fidelity and absolute counterfactual accuracy after controlling for intervention magnitude. A fresh protocol should specify the full k curve and an architecture-by-treatment-by-budget contrast in advance. Do not choose only k4 because it supports the original story.

## Reproduce

From the repository root: `bash review_robustness.sh`. This reads existing uploaded results and regenerates this report, the plots, component rows, paired tables and audit. It performs no new model forwards. The original GPU runner remains `bash run_gpu.sh main`; rerunning against its original local output skips completed checkpoints.
''')
print('Saved:',OUT/'REPORT.md')
print('Assessment: U is now decomposed into selected and random fidelity; matched-control analysis remains separate.')
