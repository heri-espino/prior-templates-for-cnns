from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS=[
 'fidelity','random_fidelity','energy_matched_fidelity',
 'causal_usefulness_random','causal_usefulness_energy',
 'cf_accuracy','random_cf_accuracy','energy_matched_cf_accuracy',
]


def paired_interval(delta):
 delta=delta.dropna();n=len(delta)
 if n<2:return None
 mean=delta.mean();err=t.ppf(.975,n-1)*delta.std()/np.sqrt(n)
 return n,mean,mean-err,mean+err


def summarize(root):
 root=Path(root);dest=root/'analysis';dest.mkdir(exist_ok=True)
 rows=[];matching=[];checks=[]
 for p in sorted((root/'runs').glob('*/*/block*/*/*.json')):
  v=json.loads(p.read_text())
  identity={k:v[k] for k in ['task','architecture','block','condition','epoch']}
  rows.extend(dict(**identity,**x,test_acc=v['test_acc']) for x in v['rows'])
  matching.extend(dict(**identity,**x) for x in v['matching'])
  comp=v.get('stage_c_comparison')
  checks.append(dict(
   **identity,
   selected_fidelity_difference=None if comp is None else comp['maximum_abs_selected_fidelity_difference'],
   random_fidelity_difference=None if comp is None else comp['maximum_abs_random_fidelity_difference'],
   U_difference=None if comp is None else comp['maximum_abs_U_difference'],
   cf_accuracy_difference=None if comp is None else comp['maximum_abs_cf_accuracy_difference'],
   full_patch_max_error=v['full_patch_max_error'],
   noop_max_error=v['noop_max_error'],
  ))
 df=pd.DataFrame(rows);mdf=pd.DataFrame(matching);cdf=pd.DataFrame(checks)
 df.to_csv(dest/'metrics.csv',index=False);mdf.to_csv(dest/'matching_diagnostics.csv',index=False);cdf.to_csv(dest/'stage_c_consistency.csv',index=False)

 contrasts=[]
 for key,s in df.groupby(['task','architecture','epoch','method','k']):
  a=s[s.condition=='template_release'].set_index('block')
  b=s[s.condition=='template_retention_1'].set_index('block')
  for metric in METRICS:
   ans=paired_interval(a[metric]-b[metric])
   if ans is None:continue
   n,mean,lo,hi=ans
   contrasts.append(dict(zip(['task','architecture','epoch','method','k'],key),metric=metric,n=n,mean=mean,ci_low=lo,ci_high=hi))
 con=pd.DataFrame(contrasts);con.to_csv(dest/'paired_contrasts.csv',index=False)

 # Matching quality summaries use validation-only quantities and are descriptive.
 match_summary=(mdf.groupby(['task','architecture','condition','method','k'])
  .agg(mean_relative_energy_error=('relative_energy_error','mean'),
       max_relative_energy_error=('relative_energy_error','max'),
       mean_energy_ratio=('energy_ratio','mean'),
       min_energy_ratio=('energy_ratio','min'),
       max_energy_ratio=('energy_ratio','max'))
  .reset_index())
 match_summary.to_csv(dest/'matching_quality.csv',index=False)

 # Main compositional-task figure: selected F, unmatched U, energy-matched U, absolute CF accuracy.
 fig,axes=plt.subplots(4,2,figsize=(11,14),sharex=True)
 display=[('fidelity','Δ selected F'),('causal_usefulness_random','Δ U random'),('causal_usefulness_energy','Δ U energy-matched'),('cf_accuracy','Δ selected CF accuracy')]
 for j,arch in enumerate(['TinyCNN','TwoLayerCNN']):
  for i,(metric,ylabel) in enumerate(display):
   ax=axes[i,j]
   for off,method in zip([-.1,0,.1],['contrast','auroc','validation_patch']):
    s=con[(con.task=='two_concepts')&(con.architecture==arch)&(con.metric==metric)&(con.method==method)].sort_values('k')
    if len(s)==0:continue
    ax.errorbar(np.arange(len(s))+off,s['mean'],yerr=[s['mean']-s.ci_low,s.ci_high-s['mean']],marker='o',capsize=3,label=method)
   ax.axhline(0,color='grey',ls='--');ax.set_xticks(range(4));ax.set_xticklabels([1,2,4,8]);ax.set_xlabel('Patched channels')
   if i==0:ax.set_title(arch)
   ax.set_ylabel(ylabel)
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3)
 fig.suptitle('two_concepts: release - retention, exploratory marginal 95% paired intervals')
 fig.tight_layout(rect=(0,.04,1,.97));fig.savefig(dest/'energy_matched_sensitivity.png',dpi=170);plt.close(fig)

 # Compact manuscript-ready CSV for the primary compositional comparison.
 compact=con[(con.task=='two_concepts')&con.metric.isin(['fidelity','causal_usefulness_random','causal_usefulness_energy','cf_accuracy'])].copy()
 compact.to_csv(dest/'two_concepts_key_contrasts.csv',index=False)

 expected=json.loads((root/'design.json').read_text())['jobs']
 max_stage_c={col:float(cdf[col].dropna().max()) if cdf[col].notna().any() else None for col in ['selected_fidelity_difference','random_fidelity_difference','U_difference','cf_accuracy_difference']}
 max_match=float(mdf.relative_energy_error.max()) if len(mdf) else None
 mean_match=float(mdf.relative_energy_error.mean()) if len(mdf) else None
 report=f'''# Validation patch-energy-matched control

Completed {len(checks)}/{expected} planned checkpoint evaluations. This is a post-hoc robustness analysis of existing Stage B models and test data, not an independent replication.

![Energy-matched sensitivity](energy_matched_sensitivity.png)

## What is compared

For every validation-only channel ranking and `k ∈ {{1,2,4,8}}`, the analysis reports the selected patch against two controls:

1. the original same-size random-channel baseline from Stage C;
2. eight same-size controls selected on validation data to match the selected subset's first-layer activation replacement energy.

The second baseline is designed to test whether a treatment difference in `U` can be explained by selected sets simply inducing larger or smaller activation replacements than arbitrary same-size sets. It does not control every possible intervention property.

## Matching quality

Across all saved matching records, mean relative validation-energy mismatch is {mean_match if mean_match is not None else 'NA'} and maximum mismatch is {max_match if max_match is not None else 'NA'}. Inspect `matching_quality.csv` and `matching_diagnostics.csv`; poor matching in any setting must be reported rather than treating the baseline as successfully matched by definition.

## Consistency with the original Stage C evaluator

Maximum absolute differences when recomputing the unchanged Stage C quantities are:

- selected fidelity: {max_stage_c['selected_fidelity_difference']}
- random fidelity: {max_stage_c['random_fidelity_difference']}
- original U: {max_stage_c['U_difference']}
- selected counterfactual accuracy: {max_stage_c['cf_accuracy_difference']}

These checks validate continuity of the old ranking/patching path; they do not make the new energy-matched analysis independent.

## Interpretation

Use `paired_contrasts.csv` and `two_concepts_key_contrasts.csv` to distinguish four questions:

- Does the release-minus-retention difference in **selected fidelity** vary with channel budget?
- Is the original `ΔU_random` pattern driven partly by the unmatched random baseline?
- Does the pattern survive as `ΔU_energy` after validation energy matching?
- Does absolute selected-patch counterfactual accuracy tell the same story?

A sign change in `ΔU_random` alone is not evidence that causal information became more distributed. If `ΔU_energy` differs qualitatively, the manuscript must describe the original finding as baseline-sensitive. If both `ΔF_selected` and `ΔU_energy` retain budget dependence, that supports a stronger—but still exploratory—functional measurement-sensitivity claim.

## Files

- `metrics.csv`: all checkpoint/method/k outcomes;
- `paired_contrasts.csv`: release minus retention paired block intervals for every metric;
- `matching_diagnostics.csv`: every selected/control validation energy match;
- `matching_quality.csv`: grouped matching quality;
- `stage_c_consistency.csv`: comparison with unchanged Stage C quantities;
- `two_concepts_key_contrasts.csv`: compact compositional-task contrasts.
'''
 (dest/'REPORT.md').write_text(report)
 print('REPORT:',dest/'REPORT.md',flush=True)


if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('output');summarize(p.parse_args().output)
