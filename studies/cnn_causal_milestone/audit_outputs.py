"""Audit complete outputs independently of the experiment's internal checks."""
import json,hashlib,argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch

from study import ROOT,TASKS,ARCHS,CONDITIONS,Network,bank,alignment,setup,collect,pairs,digest,seed,render_object
ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args();torch.set_num_threads(1)
out=ROOT/('audit_partial.json' if args.partial else 'audit.json')
freeze=json.loads((ROOT/'design_freeze.json').read_text())
assert hashlib.sha256((ROOT/'study.py').read_bytes()).hexdigest()==freeze['study_source_sha256']
assert hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest()==freeze['protocol_sha256']
seen={};streams=[];nuisance_seeds=[];dataset_checks=[]
for task in TASKS:
 for block in range(10):
  for si,split in enumerate(['train','val','test']):
   p=ROOT/'data'/task/f'block{block:02d}'/f'{split}.npz'
   if not p.exists():
    assert args.partial;continue
   with np.load(p) as z:
    x=z['x'];y=z['y'];c=z['concepts'];meta=json.loads(p.with_suffix('.json').read_text());streams.append(meta['stream_seed'])
    assert meta['image_sha256']==digest(x)
    assert np.array_equal(np.bincount(y),np.full(4,len(y)//4))
    assert len(meta['nuisance_contexts'])==len(y)//4
    nuisance_seeds.extend(c['nuisance_seed'] for c in meta['nuisance_contexts'])
    for i,img in enumerate(x):
     h=digest(img);assert h not in seen,('duplicate image',seen.get(h),(task,block,split,i));seen[h]=(task,block,split,i)
    b,cf,g=pairs(task,len(y)//4)
    assert np.all(b//4==cf//4) and np.all(y[b]!=y[cf])
    if task=='two_concepts':assert np.all((c[b]!=c[cf]).sum(1)==1)
    # The thresholded masks omit Lanczos halos: use actual clean-signal support.
    for context in range(len(y)//4):
     md=meta['nuisance_contexts'][context];clean=[]
     for state in range(4):
      if task=='single_shape':
       arr=render_object(['line','circle','triangle','square'][state],16+md['tx'],16+md['ty'],10*md['scale'],md['theta'],md['width'])
      else:
       loc=[(8+md['tx']/2,15+md['ty']),(24+md['tx']/2,17+md['ty'])]
       if md['swapped']:loc=loc[::-1]
       aa=render_object('circle' if state&1 else 'square',*loc[0],4.5*md['scale'],md['theta'],md['width'])
       bb=render_object('triangle' if state&2 else 'line',*loc[1],4.5*md['scale'],md['theta'],md['width']);arr=np.maximum(aa,bb)
      if md['occluded']:arr[md['oy']:md['oy']+5,md['ox']:md['ox']+5]=0
      clean.append(arr)
     far=np.all(np.array(clean)==0,axis=0);ids=slice(context*4,context*4+4)
     assert far.any()
     assert np.array_equal(x[ids,0][:,far],np.repeat(x[context*4,0][None,far],4,axis=0))
    dataset_checks.append(dict(task=task,block=block,split=split,n=len(y),sha256=digest(x),matched_background_verified=True))
assert len(streams)==len(set(streams))
assert len(nuisance_seeds)==len(set(nuisance_seeds))
checks=[]
for p in sorted((ROOT/'runs').glob('*/*/block*/*/result.json')):
 r=json.loads(p.read_text());d=p.parent;cfg=json.loads((d/'config.json').read_text());history=json.loads((d/'history.json').read_text());assert len(history)==40
 W=np.load(d/'conv_final.npy');Wi=np.load(d/'conv_initial.npy');assert np.isfinite(W).all()
 assert abs(alignment(W,bank())-r['alignment'])<1e-6
 if r['condition'].startswith('frozen'):assert np.array_equal(W,Wi)
 model=Network(r['architecture']);model.load_state_dict(torch.load(d/'model.pt',map_location='cpu',weights_only=True));model.eval()
 with np.load(ROOT/'data'/r['task']/f'block{r["block"]:02d}'/'test.npz') as z:x=z['x'];y=z['y']
 logits,_,_,_=collect(model,x);assert abs(float((logits.argmax(1).numpy()==y).mean())-r['id_acc'])<1e-9
 pr=json.loads((d/'patching.json').read_text());assert pr['pair_count']==(768 if r['task']=='single_shape' else 512)
 if pr['denominator']>=1e-10:
  assert abs(pr['full']['fidelity']-1)<1e-7 and abs(pr['noop']['fidelity'])<1e-7
  assert np.isfinite(r['causal_usefulness'])
 assert abs(pr['full']['cf_accuracy']-r['id_acc'])<1e-6 # balanced incoming counterfactual states
 # Check actual full-map and no-op network forwards, independent of cached probability definitions.
 with torch.no_grad():
  h=model.features(torch.from_numpy(x[:8]));l=model.tail(h);rev=np.array([1,0,3,2,5,4,7,6]);patched=h.clone();patched[:]=h[rev]
  assert torch.allclose(model.tail(patched),l[rev],atol=2e-6)
  assert torch.equal(model.tail(h.clone()),l)
 hd=json.loads((d/'head_refit.json').read_text());vs=[f['val_ce'] for f in hd['all_fits']];assert hd['selected']==int(np.argmin(vs));assert np.isfinite(vs).all()
 # Independently recompute diagnostic test prediction from stored head weights.
 _,_,pooled,_=collect(model,x);f=hd['selected_fit'];xx=(pooled.numpy()-np.array(f['mean']))/np.array(f['sd']);xx=np.column_stack([xx,np.ones(len(xx))]);w=np.array(f['weights']);recomputed=np.sum(xx[:,:,None]*w[None,:,:],axis=1)
 assert np.isfinite(recomputed).all() and abs(float((recomputed.argmax(1)==y).mean())-r['refit_test_acc'])<1e-9
 checks.append(dict(task=r['task'],architecture=r['architecture'],condition=r['condition'],block=r['block'],checkpoint_accuracy_verified=True,alignment_verified=True,patch_identities_verified=True,head_predictions_verified=True))
if not args.partial:assert len(checks)==400 and len(dataset_checks)==60
first=ROOT.parent/'cnn_first_pass'
for rec in json.loads((first/'source_manifest.json').read_text()):assert hashlib.sha256((first/rec['path']).read_bytes()).hexdigest()==rec['working_sha256']
orig=json.loads((first/'analysis/final_verification.json').read_text());assert hashlib.sha256((first/'REPORT.md').read_bytes()).hexdigest()==orig['report_sha256']
result=dict(utc=datetime.now(timezone.utc).isoformat(),run_count=len(checks),dataset_count=len(dataset_checks),unique_images=len(seen),source_and_protocol_unchanged=True,first_pass_report_unchanged=True,dataset_checks=dataset_checks,run_checks=checks)
out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['dataset_checks','run_checks']},indent=2))
