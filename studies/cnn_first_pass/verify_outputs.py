import json,hashlib,re,zipfile
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd,numpy as np
D=pd.read_csv('analysis/per_run.csv');A=json.loads(Path('analysis/audit.json').read_text());details=json.loads(Path('analysis/details.json').read_text())
assert len(D)==28==len(A['checks'])==len(details)
assert D.groupby('suite').size().to_dict()=={'controls':10,'v3_seed0':3,'v4_lowdata':15}
assert len(list(Path('runs').glob('*/*/*/model.pt')))==28
assert np.isfinite(D[['id_acc','align_mean_best','specificity','test_acc_sample','ood_rot_acc_sample','ood_thick_acc_sample','ood_occ_acc_sample']].values).all()
assert D[['test_acc_sample','ood_rot_acc_sample','ood_thick_acc_sample','ood_occ_acc_sample']].ge(0).all().all() and D[['test_acc_sample','ood_rot_acc_sample','ood_thick_acc_sample','ood_occ_acc_sample']].le(1).all().all()
for seed in range(5):
 hashes=[x['dataset_sha256'] for x in details if x['suite']!='v3_seed0' and x['seed']==seed]
 assert len(hashes)==5 and all(x==hashes[0] for x in hashes)
for r in D.itertuples():
 p=Path('runs',r.suite,r.regime,f'seed{r.seed:02d}')
 w=np.load(p/'conv_W_init.npy').reshape(16,-1)
 if r.regime in ['random_unitnorm','frozen_random_unitnorm']:
  assert np.allclose(w.mean(1),0,atol=1e-7) and np.allclose(np.linalg.norm(w,axis=1),1,atol=1e-6)
assert hashlib.sha256(Path('cnn_original.zip').read_bytes()).hexdigest()==Path('archive_sha256.txt').read_text().split()[0]
with zipfile.ZipFile('cnn_original.zip') as z:
 changed=[n for n in z.namelist() if n.endswith('.py') and not n.startswith('__MACOSX/') and z.read(n)!=Path(n).read_bytes()]
 assert changed==['src/train.py']
s=Path('REPORT.md').read_text();assert 'inserted after' not in s and 'inserted from' not in s
for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',s):assert Path(target).is_file()
logs=[json.loads(x) for x in Path('execution.jsonl').read_text().splitlines()]
assert sum(x['returncode']==0 for x in logs)==28
out={'verified_utc':datetime.now(timezone.utc).isoformat(),'completed_training_runs':28,'successful_execution_records':28,'retained_failed_control_attempts':sum(x['returncode']!=0 for x in logs),'checks_per_checkpoint':len(A['checks']),'paired_dataset_hashes_match':True,'normalization_controls_verified':True,'changed_uploaded_sources':changed,'original_archive_hash_matches':True,'report_figures_exist':True,'report_sha256':hashlib.sha256(Path('REPORT.md').read_bytes()).hexdigest()}
Path('analysis/final_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
