"""Evaluate existing checkpoints with validation patch-energy-matched controls.

Post-hoc robustness analysis only. No training occurs in this file.
"""
import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'studies/cnn_release_experiment'))
import core

EPS=1e-12
KS=(1,2,4,8)
METHODS=('contrast','auroc','validation_patch')
N_CONTROLS=8


def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,indent=2,allow_nan=False))
 os.replace(tmp,path)


def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def collect(model,x,batch,device):
 hs=[];ls=[]
 for st in range(0,len(x),batch):
  h=model.features(torch.from_numpy(x[st:st+batch]).to(device))
  hs.append(h);ls.append(model.tail(h))
 return torch.cat(hs),torch.cat(ls)


@torch.inference_mode()
def patch(model,H,orders,k,pair,batch):
 b,c,g=pair;out=[]
 orders=torch.as_tensor(orders,device=H.device)
 for st in range(0,len(b),batch):
  bs=b[st:st+batch];cs=c[st:st+batch];gs=g[st:st+batch]
  h=H[bs].clone();ix=orders[gs,:k]
  bi=torch.arange(len(bs),device=H.device)[:,None]
  h[bi,ix]=H[cs][bi,ix]
  out.append(model.tail(h).softmax(1))
 return torch.cat(out)


def pair_tensors(task,n,device):
 return tuple(torch.as_tensor(x,device=device) for x in core.pairs(task,n//4))


def measure(prob,p0,p1,y,good):
 den=float((p1-p0).square().sum());pred=prob.argmax(1)
 return dict(
  fidelity=1-float((prob-p1).square().sum())/den if den>=1e-10 else None,
  cf_accuracy=float((pred==y).float().mean()),
  agreement=float((pred==p1.argmax(1)).float().mean()),
  cf_accuracy_both_correct=float((pred[good]==y[good]).float().mean()) if good.any() else None,
 )


def channel_replacement_energy(H,pair,n_concepts):
 """Validation-only per-concept, per-channel squared replacement energy."""
 b,c,g=pair
 out=np.zeros((n_concepts,H.shape[1]),dtype=np.float64)
 for concept in range(n_concepts):
  mask=g==concept
  d=H[c[mask]]-H[b[mask]]
  out[concept]=d.square().sum((0,2,3)).detach().cpu().numpy().astype(np.float64)
 return out


def energy_matched_orders(ranking,energy,k,n_controls=N_CONTROLS):
 """Return deterministic same-k controls matched to selected validation energy.

 Each concept is matched independently. The j-th closest candidate for every
 concept is combined into one per-concept patching order for control j.
 """
 n_concepts,n_channels=energy.shape
 controls=[np.empty((n_concepts,n_channels),dtype=np.int64) for _ in range(n_controls)]
 diagnostics=[]
 all_channels=tuple(range(n_channels))
 for concept in range(n_concepts):
  selected=tuple(sorted(int(x) for x in ranking[concept,:k]))
  assert len(set(selected))==k
  target=float(energy[concept,list(selected)].sum())
  candidates=[]
  for combo in itertools.combinations(all_channels,k):
   if combo==selected:continue
   value=float(energy[concept,list(combo)].sum())
   rel=abs(value-target)/(target+EPS)
   candidates.append((rel,combo,value))
  candidates.sort(key=lambda x:(x[0],x[1]))
  chosen=candidates[:n_controls]
  assert len(chosen)==n_controls
  for j,(rel,combo,value) in enumerate(chosen):
   remaining=[ch for ch in all_channels if ch not in combo]
   controls[j][concept]=np.array(list(combo)+remaining,dtype=np.int64)
   diagnostics.append(dict(
    concept=concept,
    control=j,
    k=k,
    selected_channels=list(selected),
    control_channels=list(combo),
    selected_energy=target,
    control_energy=value,
    relative_energy_error=float(rel),
    energy_ratio=float(value/(target+EPS)),
   ))
 for ctrl in controls:
  assert ctrl.shape==(n_concepts,n_channels)
  for concept in range(n_concepts):
   assert len(set(int(x) for x in ctrl[concept,:k]))==k
   assert tuple(sorted(int(x) for x in ctrl[concept,:k]))!=tuple(sorted(int(x) for x in ranking[concept,:k]))
 return controls,diagnostics


@torch.inference_mode()
def evaluate(model,val,test,task,block,batch,device):
 hv,lv=collect(model,val['x'],batch,device)
 ht,lt=collect(model,test['x'],batch,device)
 z=hv.amax((2,3)).cpu()
 orders,_,_=core.concept_orders(z,val['concepts'])
 nc=orders.shape[0]

 # Same validation-only rankings as the original Stage C robustness analysis.
 auc_scores=np.array([[abs(core.auc(val['concepts'][:,c],z[:,ch].numpy())-.5) for ch in range(16)] for c in range(nc)])
 auc_order=np.argsort(-auc_scores,axis=1,kind='stable')
 pv=pair_tensors(task,len(hv),device);vb,vc,vg=pv
 vp0=lv[vb].softmax(1);vp1=lv[vc].softmax(1)
 scores=np.zeros((nc,16));valid=[]
 for concept in range(nc):
  valid.append(float((vp1[vg==concept]-vp0[vg==concept]).square().sum())>=1e-10)
 for ch in range(16):
  o=np.tile(np.r_[ch,np.delete(np.arange(16),ch)],(nc,1))
  prob=patch(model,hv,o,1,pv,batch)
  for concept in range(nc):
   mask=vg==concept;den=float((vp1[mask]-vp0[mask]).square().sum())
   scores[concept,ch]=1-float((prob[mask]-vp1[mask]).square().sum())/den if valid[concept] else 0
 rankings={
  'contrast':orders,
  'auroc':auc_order,
  'validation_patch':np.argsort(-scores,axis=1,kind='stable'),
 }

 energy=channel_replacement_energy(hv,pv,nc)

 pt=pair_tensors(task,len(ht),device);b,c,g=pt
 p0=lt[b].softmax(1);p1=lt[c].softmax(1);y=c%4
 good=(lt[b].argmax(1)==b%4)&(lt[c].argmax(1)==y)
 full=patch(model,ht,orders,16,pt,batch);noop=patch(model,ht,orders,0,pt,batch)
 assert torch.allclose(full,p1,atol=2e-5,rtol=1e-4),'Full patch failed'
 assert torch.allclose(noop,p0,atol=2e-5,rtol=1e-4),'No-op patch failed'

 # Preserve the original same-size random controls exactly.
 rr=core.rng(40,block,core.TASKS.index(task))
 random_orders=[np.stack([rr.permutation(16) for _ in range(nc)]) for _ in range(N_CONTROLS)]

 rows=[];raw_random={};matching=[];raw_energy={}
 for k in KS:
  random_controls=[measure(patch(model,ht,o,k,pt,batch),p0,p1,y,good) for o in random_orders]
  raw_random[str(k)]=random_controls
  random_fidelity=np.mean([x['fidelity'] for x in random_controls]) if random_controls[0]['fidelity'] is not None else None
  random_cf=float(np.mean([x['cf_accuracy'] for x in random_controls]))

  for method,o in rankings.items():
   selected=measure(patch(model,ht,o,k,pt,batch),p0,p1,y,good)
   matched_orders,diag=energy_matched_orders(o,energy,k)
   for d in diag:d['method']=method
   matching.extend(diag)
   energy_controls=[measure(patch(model,ht,co,k,pt,batch),p0,p1,y,good) for co in matched_orders]
   raw_energy[f'{method}:{k}']=energy_controls
   energy_fidelity=np.mean([x['fidelity'] for x in energy_controls]) if energy_controls[0]['fidelity'] is not None else None
   energy_cf=float(np.mean([x['cf_accuracy'] for x in energy_controls]))
   available=method!='validation_patch' or all(valid)
   rows.append(dict(
    method=method,k=k,selection_valid=available,
    **selected,
    random_fidelity=float(random_fidelity) if random_fidelity is not None else None,
    random_cf_accuracy=random_cf,
    causal_usefulness_random=selected['fidelity']-float(random_fidelity) if available and random_fidelity is not None else None,
    energy_matched_fidelity=float(energy_fidelity) if energy_fidelity is not None else None,
    energy_matched_cf_accuracy=energy_cf,
    causal_usefulness_energy=selected['fidelity']-float(energy_fidelity) if available and energy_fidelity is not None else None,
   ))

 return dict(
  test_acc=float((lt.argmax(1).cpu().numpy()==test['y']).mean()),
  rows=rows,
  rankings={k:v.tolist() for k,v in rankings.items()},
  validation_patch_scores=scores.tolist(),
  validation_patch_defined=valid,
  validation_channel_replacement_energy=energy.tolist(),
  matching=matching,
  random=raw_random,
  energy_matched=raw_energy,
  denominator=float((p1-p0).square().sum()),
  pair_count=len(b),
  full_patch_max_error=float((full-p1).abs().max()),
  noop_max_error=float((noop-p0).abs().max()),
 )


def compare_stage_c(result,task,arch,block,condition,epoch):
 p=ROOT/'results/patch_robustness_gpu_001/runs'/task/arch/f'block{block:04d}'/condition/f'epoch_{epoch:04d}.json'
 if not p.exists():return None
 old=json.loads(p.read_text())
 old_rows={(x['method'],x['k']):x for x in old['rows']}
 diffs=[]
 for x in result['rows']:
  y=old_rows[(x['method'],x['k'])]
  diffs.append(dict(
   method=x['method'],k=x['k'],
   selected_fidelity=x['fidelity']-y['fidelity'],
   random_fidelity=x['random_fidelity']-y['random_fidelity'],
   U_random=x['causal_usefulness_random']-y['causal_usefulness'],
   cf_accuracy=x['cf_accuracy']-y['cf_accuracy'],
  ))
 return dict(
  maximum_abs_selected_fidelity_difference=max(abs(x['selected_fidelity']) for x in diffs),
  maximum_abs_random_fidelity_difference=max(abs(x['random_fidelity']) for x in diffs),
  maximum_abs_U_difference=max(abs(x['U_random']) for x in diffs),
  maximum_abs_cf_accuracy_difference=max(abs(x['cf_accuracy']) for x in diffs),
  rows=diffs,
 )


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',default=str(ROOT/'results/retention_release_001'))
 p.add_argument('--output',default=str(ROOT/'outputs/patch_energy_control'))
 p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
 p.add_argument('--batch-size',type=int,default=64)
 p.add_argument('--epochs',nargs='+',type=int,default=[200])
 p.add_argument('--smoke',action='store_true')
 args=p.parse_args()
 if args.batch_size<1:p.error('batch-size must be positive')
 if args.device=='cuda' and not torch.cuda.is_available():
  raise SystemExit('CUDA unavailable: activate CUDA-enabled PyTorch. No CPU fallback is used for the recorded GPU analysis.')
 torch.set_num_threads(2)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device(args.device);inp=Path(args.input).resolve();out=Path(args.output).resolve()
 if out==inp or inp in out.parents:raise SystemExit('Output must be outside imported input tree.')
 design=json.loads((inp/'design.json').read_text())
 for name,h in design['source_hashes'].items():
  assert sha(ROOT/'studies/cnn_release_experiment'/name)==h,'Stage B source provenance mismatch'

 jobs=[]
 for task in design['tasks']:
  for arch in design['architectures']:
   for block in range(design['start_block'],design['start_block']+design['blocks']):
    for condition in ['template_retention_1','template_release']:
     for epoch in args.epochs:
      run=inp/'runs'/task/arch/f'block{block:04d}'/condition
      cp=run/f'epoch_{epoch:04d}.pt'
      jobs.append((task,arch,block,condition,epoch,run,cp))
 if args.smoke:jobs=jobs[:1]

 inputs={}
 for task,arch,block,condition,epoch,run,cp in jobs:
  if not cp.exists():raise SystemExit(f'Missing checkpoint: {cp}')
  inputs[str(cp.relative_to(inp))]=sha(cp)
  for split in ['val','test']:
   dp=inp/'data'/task/f'block{block:02d}'/f'{split}.npz'
   inputs[str(dp.relative_to(inp))]=sha(dp)

 config=dict(
  input=str(inp),epochs=args.epochs,smoke=args.smoke,batch_size=args.batch_size,
  device=args.device,torch=torch.__version__,numpy=np.__version__,
  gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
  source_hash=sha(Path(__file__)),core_hash=sha(Path(core.__file__)),
  protocol_hash=sha(Path(__file__).with_name('PROTOCOL.md')),inputs=inputs,jobs=len(jobs),
  k=list(KS),selection_methods=list(METHODS),controls=N_CONTROLS,matching='validation replacement energy',
 )
 if (out/'design.json').exists():
  assert json.loads((out/'design.json').read_text())==config,'Resume configuration/input changed: choose a new output folder.'
 else:write(out/'design.json',config)

 print('DEVICE:',device,config['gpu'],'| checkpoint evaluations:',len(jobs),flush=True)
 for num,(task,arch,block,condition,epoch,run,cp) in enumerate(jobs,1):
  target=out/'runs'/task/arch/f'block{block:04d}'/condition/f'epoch_{epoch:04d}.json'
  if target.exists():print(f'SKIP {num}/{len(jobs)}',flush=True);continue
  model=core.Network(arch)
  saved=torch.load(cp,map_location='cpu',weights_only=True)
  model.load_state_dict(saved['model']);model.to(device).eval()
  data={}
  for split in ['val','test']:
   with np.load(inp/'data'/task/f'block{block:02d}'/f'{split}.npz') as z:
    data[split]={k:z[k] for k in ['x','y','concepts']}
  result=evaluate(model,data['val'],data['test'],task,block,args.batch_size,device)
  result['stage_c_comparison']=compare_stage_c(result,task,arch,block,condition,epoch)
  result.update(task=task,architecture=arch,block=block,condition=condition,epoch=epoch)
  write(target,result)
  comp=result['stage_c_comparison']
  msg='' if comp is None else f"; max Stage-C selected-F delta={comp['maximum_abs_selected_fidelity_difference']:.3g}"
  print(f'DONE {num}/{len(jobs)} {task} {arch} {condition}{msg}',flush=True)
  del model,saved

 from summarize import summarize
 summarize(out)


if __name__=='__main__':main()
