"""GPU checkpoint robustness evaluation; original training evidence is read-only."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT/'studies/cnn_release_experiment'))
import core

def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(tmp,path)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.inference_mode()
def collect(model,x,batch,device):
 hs=[];ls=[]
 for st in range(0,len(x),batch):
  h=model.features(torch.from_numpy(x[st:st+batch]).to(device));hs.append(h);ls.append(model.tail(h))
 return torch.cat(hs),torch.cat(ls)

@torch.inference_mode()
def patch(model,H,orders,k,pair,batch):
 b,c,g=pair;out=[];orders=torch.as_tensor(orders,device=H.device)
 for st in range(0,len(b),batch):
  bs=b[st:st+batch];cs=c[st:st+batch];gs=g[st:st+batch]
  h=H[bs].clone();ix=orders[gs,:k];bi=torch.arange(len(bs),device=H.device)[:,None]
  h[bi,ix]=H[cs][bi,ix];out.append(model.tail(h).softmax(1))
 return torch.cat(out)

def pair_tensors(task,n,device):return tuple(torch.as_tensor(x,device=device) for x in core.pairs(task,n//4))
def measure(prob,p0,p1,y,good):
 den=float((p1-p0).square().sum());pred=prob.argmax(1)
 return dict(fidelity=1-float((prob-p1).square().sum())/den if den>=1e-10 else None,cf_accuracy=float((pred==y).float().mean()),agreement=float((pred==p1.argmax(1)).float().mean()),cf_accuracy_both_correct=float((pred[good]==y[good]).float().mean()) if good.any() else None)

@torch.inference_mode()
def evaluate(model,val,test,task,block,batch,device):
 hv,lv=collect(model,val['x'],batch,device);ht,lt=collect(model,test['x'],batch,device)
 z=hv.amax((2,3)).cpu();orders,_,_=core.concept_orders(z,val['concepts']);nc=orders.shape[0]
 # AUROC ranking uses validation labels only, independent of the classifier.
 auc_scores=np.array([[abs(core.auc(val['concepts'][:,c],z[:,ch].numpy())-.5) for ch in range(16)] for c in range(nc)])
 auc_order=np.argsort(-auc_scores,axis=1,kind='stable')
 # Alternative intervention-based ranking: individual channels ranked by validation CF fidelity within concept.
 pv=pair_tensors(task,len(hv),device);b,c,g=pv;p0=lv[b].softmax(1);p1=lv[c].softmax(1)
 scores=np.zeros((nc,16));valid=[]
 for concept in range(nc):valid.append(float((p1[g==concept]-p0[g==concept]).square().sum())>=1e-10)
 for ch in range(16):
  o=np.tile(np.r_[ch,np.delete(np.arange(16),ch)],(nc,1));prob=patch(model,hv,o,1,pv,batch)
  for concept in range(nc):
   mask=g==concept;den=float((p1[mask]-p0[mask]).square().sum())
   scores[concept,ch]=1-float((prob[mask]-p1[mask]).square().sum())/den if valid[concept] else 0
 rankings={'contrast':orders,'auroc':auc_order,'validation_patch':np.argsort(-scores,axis=1,kind='stable')}
 pt=pair_tensors(task,len(ht),device);b,c,g=pt;p0=lt[b].softmax(1);p1=lt[c].softmax(1);y=c%4;good=(lt[b].argmax(1)==b%4)&(lt[c].argmax(1)==y)
 full=patch(model,ht,orders,16,pt,batch);noop=patch(model,ht,orders,0,pt,batch)
 assert torch.allclose(full,p1,atol=2e-5,rtol=1e-4),'Full patch failed'
 assert torch.allclose(noop,p0,atol=2e-5,rtol=1e-4),'No-op patch failed'
 rr=core.rng(40,block,core.TASKS.index(task));random=[np.stack([rr.permutation(16) for _ in range(nc)]) for _ in range(8)]
 rows=[];raw_random={}
 for k in [1,2,4,8]:
  controls=[measure(patch(model,ht,o,k,pt,batch),p0,p1,y,good) for o in random];raw_random[str(k)]=controls
  rf=np.mean([x['fidelity'] for x in controls]) if controls[0]['fidelity'] is not None else None
  for method,o in rankings.items():
   v=measure(patch(model,ht,o,k,pt,batch),p0,p1,y,good)
   available=method!='validation_patch' or all(valid)
   rows.append(dict(method=method,k=k,selection_valid=available,**v,random_fidelity=float(rf) if rf is not None else None,random_cf_accuracy=float(np.mean([x['cf_accuracy'] for x in controls])),causal_usefulness=v['fidelity']-float(rf) if available and rf is not None else None))
 return dict(test_acc=float((lt.argmax(1).cpu().numpy()==test['y']).mean()),rows=rows,rankings={k:v.tolist() for k,v in rankings.items()},validation_patch_scores=scores.tolist(),validation_patch_defined=valid,random=raw_random,denominator=float((p1-p0).square().sum()),pair_count=len(b),full_patch_max_error=float((full-p1).abs().max()),noop_max_error=float((noop-p0).abs().max()))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default=str(ROOT/'results/retention_release_001'));p.add_argument('--output',default=str(ROOT/'outputs/patch_robustness'))
 p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--batch-size',type=int,default=64);p.add_argument('--epochs',nargs='+',type=int,default=[200]);p.add_argument('--smoke',action='store_true');args=p.parse_args()
 if args.batch_size<1:p.error('batch-size must be positive')
 if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA unavailable: activate a CUDA-enabled PyTorch environment and verify nvidia-smi. No CPU fallback was used.')
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device(args.device);inp=Path(args.input).resolve();out=Path(args.output).resolve()
 if out==inp or inp in out.parents:raise SystemExit('Output must be outside the imported input tree.')
 design=json.loads((inp/'design.json').read_text())
 for name,h in design['source_hashes'].items():assert sha(ROOT/'studies/cnn_release_experiment'/name)==h,'Source provenance mismatch'
 jobs=[]
 for task in design['tasks']:
  for arch in design['architectures']:
   for block in range(design['start_block'],design['start_block']+design['blocks']):
    for condition in ['template_retention_1','template_release']:
     for epoch in args.epochs:
      run=inp/'runs'/task/arch/f'block{block:04d}'/condition;cp=run/f'epoch_{epoch:04d}.pt'
      jobs.append((task,arch,block,condition,epoch,run,cp))
 if args.smoke:jobs=jobs[:1]
 inputs={}
 for task,arch,block,condition,epoch,run,cp in jobs:
  if not cp.exists():raise SystemExit(f'Missing checkpoint: {cp}')
  inputs[str(cp.relative_to(inp))]=sha(cp)
  for split in ['val','test']:
   dp=inp/'data'/task/f'block{block:02d}'/f'{split}.npz';inputs[str(dp.relative_to(inp))]=sha(dp)
 config=dict(input=str(inp),epochs=args.epochs,smoke=args.smoke,batch_size=args.batch_size,device=args.device,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,source_hash=sha(Path(__file__)),core_hash=sha(Path(core.__file__)),protocol_hash=sha(Path(__file__).with_name('PROTOCOL.md')),inputs=inputs,jobs=len(jobs))
 if (out/'design.json').exists():assert json.loads((out/'design.json').read_text())==config,'Resume configuration/input changed: choose a new output folder'
 else:write(out/'design.json',config)
 print('DEVICE:',device,config['gpu'],'| checkpoint evaluations:',len(jobs),flush=True)
 for num,(task,arch,block,condition,epoch,run,cp) in enumerate(jobs,1):
  target=out/'runs'/task/arch/f'block{block:04d}'/condition/f'epoch_{epoch:04d}.json'
  if target.exists():print(f'SKIP {num}/{len(jobs)}',flush=True);continue
  model=core.Network(arch);saved=torch.load(cp,map_location='cpu',weights_only=True);model.load_state_dict(saved['model']);model.to(device).eval()
  data={}
  for split in ['val','test']:
   with np.load(inp/'data'/task/f'block{block:02d}'/f'{split}.npz') as z:data[split]={k:z[k] for k in ['x','y','concepts']}
  result=evaluate(model,data['val'],data['test'],task,block,args.batch_size,device)
  old=json.loads((run/'evaluations'/f'epoch_{epoch:04d}'/'result.json').read_text());contrast=next(x for x in result['rows'] if x['method']=='contrast' and x['k']==4)
  result['legacy_difference']=dict(test_acc=result['test_acc']-old['id_acc'],causal_usefulness=contrast['causal_usefulness']-old['causal_usefulness'] if contrast['causal_usefulness'] is not None and old['causal_usefulness'] is not None else None)
  result.update(task=task,architecture=arch,block=block,condition=condition,epoch=epoch)
  write(target,result);print(f'DONE {num}/{len(jobs)} {task} {arch} {condition}; legacy ΔU={result["legacy_difference"]["causal_usefulness"]}',flush=True)
  del model,saved
 from summarize import summarize
 summarize(out)
if __name__=='__main__':main()
