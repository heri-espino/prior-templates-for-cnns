"""Independent checkpoint audit for the Stage B manuscript evidence.

This file deliberately does not import the original Stage B metric implementation.
Stored result.json files are read only after each metric has been recomputed.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.stats import rankdata
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'results/retention_release_001'
OUT=ROOT/'analysis/checkpoint_audit'
MASTER=2026090617
TASKS=['single_shape','two_concepts']
ARCHS=['TinyCNN','TwoLayerCNN']
CONDITIONS=['template_retention_1','template_release']
BLOCKS=[2000,2009]
EPOCHS=[80,200]

TOL={
 'id_acc':1e-12,
 'alignment':1e-6,
 'semantic_auc':1e-9,
 'localization_iou':1e-9,
 'fidelity_selected':2e-6,
 'fidelity_random':2e-6,
 'causal_usefulness':2e-6,
 'cf_accuracy':1e-12,
 'cf_accuracy_random':1e-12,
}


def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rng(*keys):
 return np.random.default_rng(np.random.SeedSequence([MASTER,*map(int,keys)]))


def normalize(x):
 x=np.asarray(x,dtype=np.float64)
 x=x-x.mean(axis=(-2,-1),keepdims=True)
 den=np.sqrt((x*x).sum(axis=(-2,-1),keepdims=True))
 return x/den


def bank():
 ax=np.arange(-4,5);x,y=np.meshgrid(ax,ax);g=np.exp(-(x*x+y*y)/8);out=[]
 for i in range(8):
  a=i*np.pi/8;out.append((x*np.cos(a)+y*np.sin(a))*g)
 for i in range(4):
  a=i*np.pi/2;u=x*np.cos(a)+y*np.sin(a);v=-x*np.sin(a)+y*np.cos(a)
  d1=v*v+np.minimum(u,0)**2;d2=u*u+np.minimum(v,0)**2
  out.append(np.exp(-np.minimum(d1,d2)/(2*.65**2))*g)
 rad=np.sqrt(x*x+y*y)
 for r in [1.5,2.,2.5,3.]:
  out.append(np.exp(-(rad-r)**2/2)-np.exp(-(rad-r-1)**2/2))
 return normalize(np.stack(out)).astype(np.float32)


def alignment(weights,templates):
 a=normalize(weights[:,0]).reshape(16,-1)
 b=normalize(templates).reshape(16,-1)
 sim=np.einsum('ik,jk->ij',a,b,optimize=False)
 return float(sim.max(1).mean())


class AuditNetwork(nn.Module):
 def __init__(self,arch):
  super().__init__()
  self.arch=arch
  self.conv=nn.Conv2d(1,16,9,padding=4,bias=False)
  self.conv2=nn.Conv2d(16,16,3,padding=1,bias=False) if arch=='TwoLayerCNN' else None
  self.classifier=nn.Linear(16,4)
 def features(self,x):
  return F.relu(self.conv(x))
 def tail_features(self,h):
  if self.conv2 is not None:h=F.relu(self.conv2(h))
  return h.amax((2,3))
 def tail(self,h):
  return self.classifier(self.tail_features(h))
 def forward(self,x):
  return self.tail(self.features(x))


@torch.inference_mode()
def collect(model,x,batch=128):
 """Return first-layer maps, first-layer global-max activations, and logits.

 Stage B concept ranking/AUROC is explicitly defined on the first-layer
 representation even for TwoLayerCNN. Keeping this quantity distinct from the
 downstream pooled tail representation is part of the audit specification.
 """
 maps=[];first_pooled=[];logits=[]
 for st in range(0,len(x),batch):
  h=model.features(torch.from_numpy(x[st:st+batch]))
  z=h.amax((2,3))
  p=model.tail_features(h)
  maps.append(h);first_pooled.append(z);logits.append(model.classifier(p))
 return torch.cat(maps),torch.cat(first_pooled),torch.cat(logits)


def auc(y,score):
 y=np.asarray(y,dtype=bool);n=int(y.sum());m=len(y)-n
 if not n or not m:return float('nan')
 ranks=rankdata(np.asarray(score))
 return float((ranks[y].sum()-n*(n+1)/2)/(n*m))


def concept_orders(z,concepts):
 z=np.asarray(z,dtype=np.float64)
 sd=z.std(0)+1e-6
 effects=np.stack([(z[concepts[:,i]==1].mean(0)-z[concepts[:,i]==0].mean(0))/sd for i in range(concepts.shape[1])])
 orders=np.argsort(-np.abs(effects),axis=1,kind='stable')
 signs=np.where(effects>=0,1.,-1.)
 return orders,signs,effects


def pairs(task,nctx):
 base=[];cf=[];concept=[]
 for j in range(nctx):
  for s in range(4):
   targets=[t for t in range(4) if t!=s] if task=='single_shape' else [s^1,s^2]
   for target in targets:
    base.append(j*4+s);cf.append(j*4+target)
    concept.append(target if task=='single_shape' else (0 if (s^target)==1 else 1))
 return np.array(base),np.array(cf),np.array(concept)


@torch.inference_mode()
def patch_prob(model,H,orders,k,pair,batch=128):
 b,c,g=pair;out=[]
 for st in range(0,len(b),batch):
  bs=b[st:st+batch];cs=c[st:st+batch];gs=g[st:st+batch]
  h=H[bs].clone()
  ix=torch.from_numpy(orders[gs,:k].copy())
  bi=torch.arange(len(bs))[:,None]
  h[bi,ix]=H[cs][bi,ix]
  out.append(model.tail(h).softmax(1))
 return torch.cat(out)


def patch_metrics(model,H,logits,orders,task,block):
 b,c,g=pairs(task,len(H)//4)
 p0=logits[b].softmax(1);p1=logits[c].softmax(1)
 labels=torch.tensor(c%4)
 den=float((p1-p0).square().sum())
 if den<1e-10:raise RuntimeError('Undefined fidelity denominator in planned audit checkpoint')
 good=(logits[b].argmax(1)==torch.tensor(b%4))&(logits[c].argmax(1)==labels)
 def measure(prob):
  pred=prob.argmax(1)
  return dict(
   fidelity=1-float((prob-p1).square().sum())/den,
   cf_accuracy=float((pred==labels).float().mean()),
   agreement=float((pred==p1.argmax(1)).float().mean()),
   cf_accuracy_both_correct=float((pred[good]==labels[good]).float().mean()) if good.any() else None,
  )
 selected=measure(patch_prob(model,H,orders,4,(b,c,g)))
 random_generator=rng(40,block,TASKS.index(task))
 random_orders=[np.stack([random_generator.permutation(16) for _ in range(orders.shape[0])]) for _ in range(8)]
 random_values=[measure(patch_prob(model,H,o,4,(b,c,g))) for o in random_orders]
 full=patch_prob(model,H,orders,16,(b,c,g));noop=patch_prob(model,H,orders,0,(b,c,g))
 full_error=float((full-p1).abs().max());noop_error=float((noop-p0).abs().max())
 return dict(
  fidelity_selected=selected['fidelity'],
  fidelity_random=float(np.mean([x['fidelity'] for x in random_values])),
  causal_usefulness=selected['fidelity']-float(np.mean([x['fidelity'] for x in random_values])),
  cf_accuracy=selected['cf_accuracy'],
  cf_accuracy_random=float(np.mean([x['cf_accuracy'] for x in random_values])),
  full_patch_max_error=full_error,
  noop_max_error=noop_error,
  denominator=den,
 )


def load_split(task,block,split):
 path=EVIDENCE/'data'/task/f'block{block:02d}'/f'{split}.npz'
 with np.load(path) as z:
  return {k:z[k] for k in ['x','y','concepts','masks']},path


def recompute(task,arch,block,condition,epoch):
 run=EVIDENCE/'runs'/task/arch/f'block{block:04d}'/condition
 checkpoint=run/f'epoch_{epoch:04d}.pt'
 stored_path=run/'evaluations'/f'epoch_{epoch:04d}'/'result.json'
 val,val_path=load_split(task,block,'val');test,test_path=load_split(task,block,'test')
 saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
 model=AuditNetwork(arch);model.load_state_dict(saved['model']);model.eval()
 hv,zv,lv=collect(model,val['x']);ht,zt,lt=collect(model,test['x'])
 orders,signs,effects=concept_orders(zv.numpy(),val['concepts'])

 # Independent semantic AUROC on first-layer global-max activations.
 sem=[]
 for concept in range(orders.shape[0]):
  ch=int(orders[concept,0])
  sem.append(auc(test['concepts'][:,concept],zt[:,ch].numpy()*signs[concept,ch]))

 # Independent validation-selected localization.
 thrs=np.quantile(hv.numpy(),.95,axis=(0,2,3));ious=[]
 for concept in range(orders.shape[0]):
  vp=val['concepts'][:,concept]==1;vm=val['masks'][vp,concept]
  loc=[]
  for ch in range(16):
   pred=hv[vp,ch].numpy()>thrs[ch]
   union=np.logical_or(pred,vm).sum()
   loc.append(np.logical_and(pred,vm).sum()/union if union else 0.)
  ch=int(np.argmax(loc));positive=test['concepts'][:,concept]==1
  pred=ht[positive,ch].numpy()>float(thrs[ch]);truth=test['masks'][positive,concept]
  union=np.logical_or(pred,truth).sum();ious.append(float(np.logical_and(pred,truth).sum()/union) if union else 0.)

 patch=patch_metrics(model,ht,lt,orders,task,block)
 weights=model.conv.weight.detach().numpy()
 recomputed=dict(
  id_acc=float((lt.argmax(1).numpy()==test['y']).mean()),
  alignment=alignment(weights,bank()),
  semantic_auc=float(np.mean(sem)),
  localization_iou=float(np.mean(ious)),
  fidelity_selected=patch['fidelity_selected'],
  fidelity_random=patch['fidelity_random'],
  causal_usefulness=patch['causal_usefulness'],
  cf_accuracy=patch['cf_accuracy'],
  cf_accuracy_random=patch['cf_accuracy_random'],
 )
 stored=json.loads(stored_path.read_text())
 diffs={k:abs(recomputed[k]-stored[k]) for k in recomputed}
 passes={k:bool(diffs[k]<=TOL[k]) for k in recomputed}
 passes['full_patch_identity']=bool(patch['full_patch_max_error']<=2e-5)
 passes['noop_identity']=bool(patch['noop_max_error']<=2e-5)
 return dict(
  task=task,architecture=arch,block=block,condition=condition,epoch=epoch,
  recomputed=recomputed,stored={k:stored[k] for k in recomputed},abs_diff=diffs,passes=passes,
  full_patch_max_error=patch['full_patch_max_error'],noop_max_error=patch['noop_max_error'],
  checkpoint_sha256=sha(checkpoint),val_sha256=sha(val_path),test_sha256=sha(test_path),stored_result_sha256=sha(stored_path),
 )


def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 OUT.mkdir(parents=True,exist_ok=True)
 records=[];flat=[]
 for task in TASKS:
  for arch in ARCHS:
   for condition in CONDITIONS:
    for block in BLOCKS:
     for epoch in EPOCHS:
      print('AUDIT',task,arch,condition,block,epoch,flush=True)
      record=recompute(task,arch,block,condition,epoch);records.append(record)
      row={k:record[k] for k in ['task','architecture','block','condition','epoch']}
      for metric in TOL:
       row[f'{metric}_recomputed']=record['recomputed'][metric]
       row[f'{metric}_stored']=record['stored'][metric]
       row[f'{metric}_abs_diff']=record['abs_diff'][metric]
       row[f'{metric}_pass']=record['passes'][metric]
      row['full_patch_max_error']=record['full_patch_max_error'];row['noop_max_error']=record['noop_max_error']
      row['full_patch_identity_pass']=record['passes']['full_patch_identity'];row['noop_identity_pass']=record['passes']['noop_identity']
      flat.append(row)

 df=pd.DataFrame(flat);df.to_csv(OUT/'audit_rows.csv',index=False)
 maxima={metric:float(df[f'{metric}_abs_diff'].max()) for metric in TOL}
 failures=[]
 for rec in records:
  for name,ok in rec['passes'].items():
   if not ok:failures.append(dict(task=rec['task'],architecture=rec['architecture'],block=rec['block'],condition=rec['condition'],epoch=rec['epoch'],check=name))
 provenance=dict(
  python=sys.version,torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
  audit_source_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).with_name('AUDIT_PROTOCOL.md')),
  stage_b_design_sha256=sha(EVIDENCE/'design.json'),planned_checkpoints=len(records),max_abs_differences=maxima,
  failures=failures,all_passed=not failures,
 )
 (OUT/'audit.json').write_text(json.dumps(dict(provenance=provenance,records=records),indent=2)+'\n')
 lines=['# Independent checkpoint audit','',f"Planned and completed checkpoint evaluations: **{len(records)}**.",'',
        'This audit independently reimplemented the central metric computations and read stored scalar results only after recomputation.','',
        f"Overall status: **{'PASS' if not failures else 'REVIEW REQUIRED'}**.",'','## Maximum absolute discrepancies','',
        '| Metric | Maximum absolute difference | Tolerance |','|---|---:|---:|']
 for metric in TOL:lines.append(f"| `{metric}` | {maxima[metric]:.12g} | {TOL[metric]:.12g} |")
 lines+=['','## Identity checks','',f"Maximum full-patch probability error: {float(df.full_patch_max_error.max()):.12g}.",f"Maximum no-op probability error: {float(df.noop_max_error.max()):.12g}.",'']
 if failures:
  lines+=['## Failures requiring reconciliation','']
  for x in failures:lines.append(f"- {x['task']} / {x['architecture']} / block {x['block']} / {x['condition']} / epoch {x['epoch']}: `{x['check']}`")
 else:
  lines+=['## Audit conclusion','',
          'All sampled central metrics and patch identities are within the tolerances fixed in `AUDIT_PROTOCOL.md`. This supports describing the sampled Stage B checkpoint evidence as independently recomputed. It does not validate unsampled checkpoints or broader scientific claims.']
 (OUT/'AUDIT_REPORT.md').write_text('\n'.join(lines)+'\n')
 print('AUDIT STATUS:',provenance['all_passed'],flush=True)
 print('REPORT:',OUT/'AUDIT_REPORT.md',flush=True)
 if failures:raise SystemExit(1)


if __name__=='__main__':main()
