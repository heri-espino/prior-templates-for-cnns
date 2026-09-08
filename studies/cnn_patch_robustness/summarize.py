from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def summarize(root):
 root=Path(root);dest=root/'analysis';dest.mkdir(exist_ok=True);rows=[];checks=[]
 for p in sorted((root/'runs').glob('*/*/block*/*/*.json')):
  v=json.loads(p.read_text());identity={k:v[k] for k in ['task','architecture','block','condition','epoch']}
  rows.extend(dict(**identity,**x,test_acc=v['test_acc']) for x in v['rows']);checks.append(dict(**identity,**v['legacy_difference']))
 df=pd.DataFrame(rows);df.to_csv(dest/'metrics.csv',index=False);pd.DataFrame(checks).to_csv(dest/'legacy_comparison.csv',index=False)
 contrasts=[]
 for key,s in df.groupby(['task','architecture','epoch','method','k']):
  a=s[s.condition=='template_release'].set_index('block');b=s[s.condition=='template_retention_1'].set_index('block')
  for metric in ['causal_usefulness','cf_accuracy']:
   delta=(a[metric]-b[metric]).dropna();n=len(delta)
   if n<2:continue
   avg=delta.mean();err=t.ppf(.975,n-1)*delta.std()/np.sqrt(n)
   contrasts.append(dict(zip(['task','architecture','epoch','method','k'],key),metric=metric,n=n,mean=avg,ci_low=avg-err,ci_high=avg+err))
 pd.DataFrame(contrasts).to_csv(dest/'paired_contrasts.csv',index=False)
 settings=list(df[['task','architecture','epoch']].drop_duplicates().itertuples(index=False,name=None))
 fig,axes=plt.subplots(1,len(settings),figsize=(5*len(settings),4),squeeze=False)
 for ax,(task,arch,epoch) in zip(axes[0],settings):
  s=df[(df.task==task)&(df.architecture==arch)&(df.epoch==epoch)]
  for (condition,method),g in s.groupby(['condition','method']):
   means=g.groupby('k').causal_usefulness.mean();ax.plot(means.index,means,marker='o',label=condition+' / '+method)
  ax.set_title(f'{task}\n{arch} epoch {epoch}');ax.set_xlabel('Patched channels');ax.set_ylabel('Mean U');ax.axhline(0,color='grey',ls=':')
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2);fig.tight_layout(rect=(0,.24,1,1));fig.savefig(dest/'robustness.png',dpi=160);plt.close(fig)
 expected=json.loads((root/'design.json').read_text())['jobs']
 (dest/'REPORT.md').write_text(f'''# Patching robustness

Completed {len(checks)}/{expected} planned checkpoint evaluations. This is a post hoc sensitivity analysis on existing models/test data, not independent replication.

![Robustness](robustness.png)

`metrics.csv` includes U, absolute counterfactual accuracy and random controls at every method/size. `paired_contrasts.csv` reports release minus constant retention with exploratory marginal paired t intervals; no multiplicity adjustment. Empty contrasts in a one-model smoke run are expected. `legacy_comparison.csv` checks original contrast/k4 against stored CPU outcomes; inspect differences before scientific interpretation. Channels are selected using validation only. Same-size random rankings are shared across conditions. Invalid selection denominators yield undefined U rather than a scientific conclusion.
''')
 print('REPORT:',dest/'REPORT.md',flush=True)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('output');summarize(p.parse_args().output)
