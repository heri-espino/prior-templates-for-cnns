"""Review uploaded robustness outcomes; no training or automatic confirmatory gate."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
R=ROOT/'results/patch_robustness_gpu_001'
rows=[];legacy=[]
files=sorted((R/'runs').glob('*/*/block*/*/*.json'));assert len(files)==80
for p in files:
 v=json.loads(p.read_text());identity={k:v[k] for k in ['task','architecture','block','condition','epoch']}
 assert len(v['rows'])==12 and v['denominator']>=1e-10
 assert v['full_patch_max_error']<2e-5 and v['noop_max_error']<2e-5
 assert all(v['validation_patch_defined'])
 rows.extend(dict(**identity,**x) for x in v['rows']);legacy.append(v['legacy_difference'])
df=pd.DataFrame(rows);assert len(df)==960 and not df.duplicated(['task','architecture','block','condition','epoch','method','k']).any()
contrasts=[]
for key,s in df.groupby(['task','architecture','method','k']):
 for metric in ['causal_usefulness','cf_accuracy']:
  a=s[s.condition=='template_release'].set_index('block')[metric];b=s[s.condition=='template_retention_1'].set_index('block')[metric];delta=(a-b).dropna();assert len(delta)==10
  mean=delta.mean();err=t.ppf(.975,9)*delta.std()/np.sqrt(10)
  contrasts.append(dict(zip(['task','architecture','method','k'],key),metric=metric,n=10,mean=mean,ci_low=mean-err,ci_high=mean+err))
c=pd.DataFrame(contrasts);c.to_csv(OUT/'paired_contrasts.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
for j,arch in enumerate(['TinyCNN','TwoLayerCNN']):
 for i,metric in enumerate(['causal_usefulness','cf_accuracy']):
  ax=axes[i,j]
  for offset,method in zip([-.1,0,.1],['contrast','auroc','validation_patch']):
   s=c[(c.task=='two_concepts')&(c.architecture==arch)&(c.metric==metric)&(c.method==method)].sort_values('k')
   ax.errorbar(np.arange(4)+offset,s['mean'],yerr=[s['mean']-s.ci_low,s.ci_high-s['mean']],marker='o',capsize=3,label=method)
  ax.axhline(0,color='grey',ls='--');ax.set_xticks(range(4));ax.set_xticklabels([1,2,4,8]);ax.set_xlabel('Patched channels');ax.set_title(arch);ax.set_ylabel('Release − constant: '+metric)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3);fig.suptitle('two_concepts: exploratory marginal 95% paired intervals (10 blocks)');fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(OUT/'measurement_sensitivity.png',dpi=170);plt.close(fig)
max_u=max(abs(v['causal_usefulness']) for v in legacy);max_acc=max(abs(v['test_acc']) for v in legacy)
(OUT/'audit.json').write_text(json.dumps(dict(checkpoints=80,method_size_rows=960,paired_blocks=10,raw_checks_passed=True,maximum_legacy_U_difference=max_u,maximum_legacy_accuracy_difference=max_acc,scope='Raw stored results and patch identities; not independent rerun of GPU forwards.'),indent=2)+'\n')
(OUT/'REPORT.md').write_text(f'''# Does the causal conclusion survive measurement changes?

All 80 GPU evaluations are present. Raw JSON yielded 960 method/size records. Maximum original-k4 U difference versus stored CPU results: {max_u:.3g}; test accuracy difference: {max_acc:.3g}. All stored full/no-op checks and validation-selection denominators passed.

![Measurement sensitivity](measurement_sensitivity.png)

## Finding

The result survives **particular** changes, not all changes. On two_concepts / TinyCNN, all three selection methods yield positive release-minus-constant U at k4 and k8, but negative differences at k1 and k2. Thus the apparent advantage is patch-budget dependent and not merely an artifact of the original contrast ranking.

For TwoLayerCNN, contrast and AUROC show negative differences at k4; validation-patch ranking gives a negative mean with an interval spanning zero. At k8, validation-patch ranking gives a positive difference. A universal claim that release improves shallow causal usefulness and harms deeper causal usefulness is therefore too broad.

These are reused checkpoints/test data, post hoc analyses, and marginal intervals without multiplicity adjustment. A sign change is descriptive evidence of measurement dependence; it does not alone establish the mechanism. Singleton-based validation-patch ranking does not optimize joint subsets.

## Decision for the conditional replication

**Defer training the proposed confirmation of the broad architecture claim.** The original condition—robustness across measurement choices—is not established. This is a scientific assessment, not a predeclared statistical pass/fail threshold.

A narrower next question is whether release changes how causal effects are distributed across channel-set sizes. A fresh annotated-task protocol should then specify the full k curve and an architecture-by-treatment-by-budget contrast in advance, with an absolute counterfactual-accuracy endpoint alongside relative U. Do not choose only k4 because it supports the original story. Before committing to new training, decide whether this narrower question is necessary for the single paper; current evidence already supports an explicitly scoped measurement-sensitivity finding.

## Reproduce

From the repository root: `bash review_robustness.sh`. This reads existing uploaded results and regenerates this report, the plot, paired tables and audit. The original GPU runner remains `bash run_gpu.sh main`; rerunning against its original local output skips completed checkpoints. No new annotated-task training is launched.
''')
print('Saved:',OUT/'REPORT.md');print('Assessment: broad causal claim depends on patch budget/selection; replication deferred.')
