"""Analyze the frozen intervention-budget confirmation design.

The primary test is fixed in PROTOCOL.md and is not selected from these outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_1samp

KS=(1,2,4,8)
METHODS=('contrast','auroc','validation_patch')
ARCHS=('TinyCNN','TwoLayerCNN')
CONDITIONS=('template_retention_1','template_release')


def sha256(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit_for(path):
 """Return the commit that most recently changed path, if Git metadata exists."""
 try:
  root=Path(__file__).resolve().parents[2]
  rel=Path(path).resolve().relative_to(root)
  return subprocess.check_output(
   ['git','log','-n','1','--format=%H','--',str(rel)],
   cwd=root,text=True,stderr=subprocess.DEVNULL,
  ).strip() or None
 except (OSError,subprocess.CalledProcessError,ValueError):
  return None


def interval(x):
 x=np.asarray(x,dtype=float);n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1))
 err=float(student_t.ppf(.975,n-1)*sd/np.sqrt(n))
 return dict(n=n,mean=mean,sd=sd,ci_low=mean-err,ci_high=mean+err)


def load_patch_rows(root):
 rows=[];checks=[]
 files=sorted(root.glob('**/patch_eval/runs/two_concepts/*/block*/*/epoch_0200.json'))
 for p in files:
  v=json.loads(p.read_text())
  ident={k:v[k] for k in ['task','architecture','block','condition','epoch']}
  rows.extend(dict(**ident,**r) for r in v['rows'])
  checks.append(dict(**ident,full_patch_max_error=v['full_patch_max_error'],noop_max_error=v['noop_max_error'],test_acc=v['test_acc']))
 return pd.DataFrame(rows),pd.DataFrame(checks),files


def load_energy_rows(root):
 rows=[];matching=[];files=[]
 for p in sorted(root.glob('**/energy_eval/runs/two_concepts/*/block*/*/epoch_0200.json')):
  files.append(p);v=json.loads(p.read_text());ident={k:v[k] for k in ['task','architecture','block','condition','epoch']}
  rows.extend(dict(**ident,**r) for r in v['rows'])
  matching.extend(dict(**ident,**m) for m in v['matching'])
 return pd.DataFrame(rows),pd.DataFrame(matching),files


def verify_design(df,files):
 expected_blocks=set(range(4000,4020))
 assert set(df.architecture)==set(ARCHS),set(df.architecture)
 assert set(df.condition)==set(CONDITIONS),set(df.condition)
 assert set(df.block)==expected_blocks,(min(df.block),max(df.block),len(set(df.block)))
 assert set(df.method)==set(METHODS)
 assert set(df.k)==set(KS)
 assert not df.duplicated(['architecture','block','condition','method','k']).any()
 assert len(df)==2*20*2*3*4,len(df)
 assert len(files)==2*20*2,len(files)


def treatment_contrasts(df,metric):
 rec=[];per_block=[]
 for arch in ARCHS:
  for method in METHODS:
   s=df[(df.architecture==arch)&(df.method==method)]
   for k in KS:
    a=s[(s.condition=='template_release')&(s.k==k)].set_index('block')[metric]
    b=s[(s.condition=='template_retention_1')&(s.k==k)].set_index('block')[metric]
    d=(a-b).sort_index();assert len(d)==20
    assert np.isfinite(d.to_numpy(dtype=float)).all(),(arch,method,k,metric)
    q=interval(d.values)
    rec.append(dict(architecture=arch,method=method,k=k,metric=metric,**q))
    for block,value in d.items():per_block.append(dict(architecture=arch,method=method,k=k,metric=metric,block=block,delta=value))
 return pd.DataFrame(rec),pd.DataFrame(per_block)


def budget_contrasts(per_block):
 rec=[];vals=[]
 s=per_block[per_block.metric=='fidelity']
 for arch in ARCHS:
  for method in METHODS:
   p=s[(s.architecture==arch)&(s.method==method)].pivot(index='block',columns='k',values='delta').sort_index()
   assert list(p.columns)==list(KS) and len(p)==20
   B=(p[[4,8]].mean(axis=1)-p[[1,2]].mean(axis=1))
   q=interval(B.values);test=ttest_1samp(B.values,0.0)
   rec.append(dict(architecture=arch,method=method,metric='budget_contrast_B',t_stat=float(test.statistic),p_two_sided=float(test.pvalue),**q))
   vals.extend(dict(architecture=arch,method=method,block=int(block),B=float(value)) for block,value in B.items())
 return pd.DataFrame(rec),pd.DataFrame(vals)


def main():
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--out',default='analysis/budget_confirmation');args=p.parse_args()
 root=Path(args.root).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
 protocol=Path(__file__).with_name('PROTOCOL.md')
 protocol_hash=sha256(protocol);protocol_commit=git_commit_for(protocol)
 analysis_hash=sha256(Path(__file__))

 patch,checks,patch_files=load_patch_rows(root);verify_design(patch,patch_files)
 assert patch.selection_valid.dtype==bool or set(patch.selection_valid.dropna().unique()).issubset({True,False})
 primary_rows=patch[(patch.architecture=='TinyCNN')&(patch.method=='contrast')]
 assert primary_rows.selection_valid.all(),'Primary contrast ranking must be defined in every planned run.'
 assert primary_rows.fidelity.notna().all(),'Primary selected fidelity is undefined in at least one planned run.'

 energy,matching,energy_files=load_energy_rows(root)
 if len(energy):verify_design(energy,energy_files)
 assert float(checks.full_patch_max_error.max())<=2e-5
 assert float(checks.noop_max_error.max())<=2e-5

 # Primary data use selected fidelity from the standard patch evaluator.
 fidelity_summary,fidelity_block=treatment_contrasts(patch,'fidelity')
 cf_summary,cf_block=treatment_contrasts(patch,'cf_accuracy')
 u_summary,u_block=treatment_contrasts(patch,'causal_usefulness')
 all_summary=pd.concat([fidelity_summary,cf_summary,u_summary],ignore_index=True)
 all_block=pd.concat([fidelity_block,cf_block,u_block],ignore_index=True)
 B,Bblock=budget_contrasts(fidelity_block)

 primary=B[(B.architecture=='TinyCNN')&(B.method=='contrast')].iloc[0]
 primary_pass=bool(primary['mean']>0 and primary['ci_low']>0 and primary['p_two_sided']<.05)

 all_summary.to_csv(out/'treatment_contrasts.csv',index=False)
 all_block.to_csv(out/'per_block_treatment_effects.csv',index=False)
 B.to_csv(out/'budget_contrasts.csv',index=False);Bblock.to_csv(out/'budget_contrasts_per_block.csv',index=False)
 checks.to_csv(out/'integrity_checks.csv',index=False)

 energy_note='Energy-matched secondary analysis was not available.'
 if len(energy):
  eu_summary,eu_block=treatment_contrasts(energy,'causal_usefulness_energy')
  ef_summary,ef_block=treatment_contrasts(energy,'fidelity')
  eu_summary.to_csv(out/'energy_U_contrasts.csv',index=False);ef_summary.to_csv(out/'energy_selected_fidelity_check.csv',index=False)
  matching.to_csv(out/'energy_matching_diagnostics.csv',index=False)
  energy_note=f"Energy-matched outputs were complete. Mean relative matching error={matching.relative_energy_error.mean():.6g}; maximum={matching.relative_energy_error.max():.6g}."

 # Compact complete primary-ranking curve.
 main_curve=all_summary[(all_summary.architecture=='TinyCNN')&(all_summary.method=='contrast')&all_summary.metric.isin(['fidelity','cf_accuracy','causal_usefulness'])]
 main_curve.to_csv(out/'primary_curve.csv',index=False)

 lines=[
 '# Prospective intervention-budget confirmation', '',
 'Protocol: `studies/cnn_budget_confirmation/PROTOCOL.md` was committed before blocks 4000–4019 were generated.', '',
 f"- protocol SHA-256: `{protocol_hash}`",
 f"- protocol commit: `{protocol_commit or 'unavailable (no Git metadata)'}`",
 f"- analysis source SHA-256: `{analysis_hash}`", '',
 f"Planned final checkpoint evaluations: **80**; loaded: **{len(patch_files)}**.", '',
 '## Primary test', '',
 'The single primary test is the pre-specified TinyCNN / contrast-ranking selected-fidelity budget contrast', '',
 r'\[B=\tfrac12[\Delta F(4)+\Delta F(8)]-\tfrac12[\Delta F(1)+\Delta F(2)].\]', '',
 f"- n = {int(primary['n'])} fresh blocks",
 f"- mean B = {primary['mean']:+.6f}",
 f"- SD = {primary['sd']:.6f}",
 f"- 95% t interval = [{primary['ci_low']:+.6f}, {primary['ci_high']:+.6f}]",
 f"- t = {primary['t_stat']:.4f}",
 f"- two-sided p = {primary['p_two_sided']:.8g}",
 f"- predeclared primary decision = **{'CONFIRMED' if primary_pass else 'NOT CONFIRMED'}**", '',
 'The decision above is computed mechanically from the frozen rule: positive mean B, lower 95% interval bound above zero, and p < 0.05.', '',
 '## Full selected-fidelity curve', '',
 '| Architecture | Ranking | k | Δ selected F | 95% interval |',
 '|---|---|---:|---:|---:|',
 ]
 for _,r in fidelity_summary.iterrows():
  lines.append(f"| {r.architecture} | {r.method} | {int(r.k)} | {r['mean']:+.4f} | [{r.ci_low:+.4f}, {r.ci_high:+.4f}] |")
 lines += ['', '## Secondary checks', '', energy_note, '',
 'All AUROC/validation-patch rankings, TwoLayerCNN results, absolute counterfactual accuracy, U, and energy-matched controls are secondary. They do not change the primary decision.', '',
 '## Integrity', '',
 f"Maximum full-patch error: {checks.full_patch_max_error.max():.3g}.",
 f"Maximum no-op error: {checks.noop_max_error.max():.3g}.", '',
 'This confirmation is independent in renderer blocks, not in task family or codebase. It does not establish a unique mechanism, human interpretability, or natural-image generalization.'
 ]
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 decision=dict(
  primary_pass=primary_pass,
  primary=primary.to_dict(),
  patch_files=len(patch_files),
  energy_files=len(energy_files),
  protocol_sha256=protocol_hash,
  protocol_commit=protocol_commit,
  analysis_sha256=analysis_hash,
 )
 (out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
 print('PRIMARY CONFIRMATION:',primary_pass)
 print('PROTOCOL COMMIT:',protocol_commit or 'unavailable')
 print('PROTOCOL SHA256:',protocol_hash)
 print('REPORT:',out/'REPORT.md')


if __name__=='__main__':main()
