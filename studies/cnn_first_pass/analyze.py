"""Supplement repository metrics with explicit, independently checked diagnostics."""
import os,json,hashlib,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.shapes import ShapesDataset,default_splits,SHAPES
from src.models.cnn import TinyCNN,ModelConfig
from src.templates.primitives import TemplateSpec,make_template_bank,template_names
from src.interpret.patching import _make_mismatched_pairs
from src.interpret.alignment import compute_alignment
from src.interpret.drift import compute_drift
from src.utils.seed import set_seed
from src.analysis.aggregate import load_run,time_to_threshold
os.chdir(Path(__file__).resolve().parent)
torch.set_num_threads(2)
smoke='--smoke' in sys.argv
report=Path('analysis_smoke' if smoke else 'analysis');report.mkdir(exist_ok=True)
T=make_template_bank(TemplateSpec()); flat=T.reshape(16,-1)
u,s,v=np.linalg.svd(flat.astype(float),full_matrices=False)
audit={'template_names':template_names(TemplateSpec()),'singular_values':s.tolist(),'rank_tol_1e-6':int((s>1e-6).sum()),'corner_edge_abs_cosines':(flat[8:12]@flat[:8].T).tolist(),'template_norms':np.linalg.norm(flat,axis=1).tolist(),'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()},'checks':[]}
rows=[];detailed=[]
cache=report/'cache';cache.mkdir(exist_ok=True)
def cachepath(path):return cache/('_'.join(path.parts[1:4])+'.json')
@torch.no_grad()
def collect(model,ds):
 ls=[];zs=[]
 for x,y in DataLoader(ds,batch_size=128):
  l,z=model(x);ls.append(l);zs.append(z)
 return torch.cat(ls),torch.cat(zs),torch.from_numpy(ds.y)
def h(ds):return hashlib.sha256(ds.x.tobytes()+ds.y.tobytes()).hexdigest()
@torch.no_grad()
def patch_stats(L,Z,Y,V,idx_s,idx_t,orders,k):
 ys,yt=Y[idx_s],Y[idx_t];lt=L[idx_t]
 ix=torch.tensor(orders[ys.numpy(),:k].copy(),dtype=torch.long)
 dz=(Z[idx_s]-Z[idx_t]).gather(1,ix)
 dl=torch.einsum('pk,cpk->pc',dz,V[:,ix])
 patched=lt+dl
 ar=torch.arange(len(ys))
 shift=dl[ar,ys]-dl[ar,yt]
 base_pred=lt.argmax(1); pred=patched.argmax(1)
 good=(base_pred==yt)&(L[idx_s].argmax(1)==ys)
 return {'positive_shift':float((shift>0).float().mean()),'mean_shift':float(shift.mean()),'prediction_changed':float((pred!=base_pred).float().mean()),'source_prediction':float((pred==ys).float().mean()),'source_prediction_both_correct':float((pred[good]==ys[good]).float().mean()) if good.any() else None,'both_correct_n':int(good.sum())}
for suite in (['v4_lowdata'] if smoke else ['v4_lowdata','v3_seed0','controls']):
 seeds=range(5) if suite!='v3_seed0' and not smoke else [0]
 for seed in seeds:
  paths=sorted(Path('runs',suite).glob(f'*/seed{seed:02d}/model.pt'))
  if smoke:paths=paths[:1]
  paths=[p for p in paths if not cachepath(p).exists() or json.loads(cachepath(p).read_text()).get('checkpoint_sha256')!=hashlib.sha256(p.read_bytes()).hexdigest()]
  if not paths:continue
  cfg=json.loads((paths[0].parent/'config.json').read_text())['args']
  splits=default_splits(seed,cfg['train_size'],cfg['val_size'],cfg['test_size'])
  ds={k:ShapesDataset(s,image_size=cfg['image_size']) for k,s in splits.items()}
  hashes={k:h(d) for k,d in ds.items()}
  for path in paths:
   d=path.parent;c=json.loads((d/'config.json').read_text());a=c['args'];regime=d.parent.name
   model=TinyCNN(ModelConfig(**c['model']),templates=T if a['regime']!='random' else None)
   model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
   r=load_run(d);met=r.metrics
   row={'suite':suite,'regime':regime,'seed':seed,**{k:met[k] for k in ['id_acc','ood_rot_acc','ood_thick_acc','ood_occ_acc']},'align_mean_best':met['alignment']['mean_best_score'],'specificity':met['conditional_ablation']['mean_specificity'],'patch_single_success':met['patching_single']['mean_success'],'patch_single_shift':met['patching_single']['mean_shift'],'patch_topk_success_at_maxk_repo':met['patching_topk']['success@maxk'],'patch_topk_mean_shift':met['patching_topk']['mean_shift'],'drift_cosine':float(r.drift_cos.mean()),'drift_l2':float(r.drift_l2.mean()),'init_kernel_norm':float(np.linalg.norm(r.conv_W_init.reshape(16,-1),axis=1).mean()),'initial_alignment':compute_alignment(torch.from_numpy(r.conv_W_init),T).summary['mean_best_score'],'val_final_repo':float(r.train_log.val_acc.iloc[-1]),'val_max_repo':float(r.train_log.val_acc.max()),'t98':time_to_threshold(r.train_log,key='val_acc',thr=.98),'t100':time_to_threshold(r.train_log,key='val_acc',thr=1.)}
   for E in [10,20,50]:
    sub=r.train_log[r.train_log.epoch<=E];row[f'auc_{E}']=float(np.trapz(sub.val_acc,sub.epoch)) if E<=r.train_log.epoch.max() else float('nan')
   outputs={k:collect(model,dset) for k,dset in ds.items() if k!='train'}
   majority=int(np.bincount(ds['train'].y,minlength=4).argmax());confusions={}
   for k,(L,Z,Y) in outputs.items():
    pred=L.argmax(1);row[f'{k}_acc_sample']=float((pred==Y).double().mean())
    row[f'{k}_majority_baseline']=float((Y==majority).double().mean())
    cm=np.zeros((4,4),dtype=int);np.add.at(cm,(Y.numpy(),pred.numpy()),1);confusions[k]=cm.tolist()
   L,Z,Y=outputs['test'];LV,ZV,YV=outputs['val'];V=model.classifier.weight.detach()
   means=torch.stack([ZV[YV==c].mean(0) for c in range(4)])
   delta=(V*means).T.numpy();orders=np.argsort(-delta,axis=0).T
   ss,tt=_make_mismatched_pairs(Y,a['num_pairs'],4,seed)
   probes={}
   rng=np.random.default_rng(seed+70123)
   random_orders=[np.stack([rng.permutation(16) for _ in range(4)]) for _ in range(20)]
   for k in [1,2,4,8,16]:
    targeted=patch_stats(L,Z,Y,V,ss,tt,orders,k)
    controls=[patch_stats(L,Z,Y,V,ss,tt,o,k) for o in random_orders]
    control={key:float(np.mean([x[key] for x in controls])) for key in targeted if targeted[key] is not None}
    probes[str(k)]={'validation_ranked':targeted,'random_ranked_mean20':control}
   row['patch_val_rank_k8']=probes['8']['validation_ranked']['positive_shift'];row['patch_random_rank_k8']=probes['8']['random_ranked_mean20']['positive_shift']
   row['patch_val_rank_k8_source_prediction']=probes['8']['validation_ranked']['source_prediction']
   row['patch_val_rank_k8_source_prediction_both_correct']=probes['8']['validation_ranked']['source_prediction_both_correct']
   # Numerical unit checks against direct classifier interventions.
   z=Z[:32].clone();z[:,0]=0
   exact=model.classifier(Z[:32])-model.classifier(z)
   assert torch.allclose(exact,Z[:32,0,None]*V[:,0][None,:],atol=2e-6)
   full=model.classifier(Z[tt]+(Z[ss]-Z[tt]))
   assert torch.allclose(full,L[ss],atol=2e-6)
   assert torch.equal(model.classifier(Z[tt]),model.classifier(Z[tt].clone()))
   # Check saved float32/BLAS cosine output using direct float64 summation.
   wf=r.conv_W_final.reshape(16,-1).astype(float);tf=T.reshape(16,-1).astype(float)
   wf-=wf.mean(1,keepdims=True);tf-=tf.mean(1,keepdims=True)
   wf/=np.linalg.norm(wf,axis=1,keepdims=True)+1e-8;tf/=np.linalg.norm(tf,axis=1,keepdims=True)+1e-8
   direct=np.sum(wf[:,None,:]*tf[None,:,:],axis=2)
   saved=np.load(d/'alignment_matrix.npy')
   assert np.isfinite(saved).all() and np.max(np.abs(saved-direct))<1e-5
   frozen='frozen' in regime
   if frozen:assert np.array_equal(r.conv_W_init,r.conv_W_final)
   # Original accuracy reproduced from saved checkpoint.
   exact_repo=float(np.mean([float((L[i:i+128].argmax(1)==Y[i:i+128]).float().mean()) for i in range(0,len(Y),128)]))
   assert abs(exact_repo-met['id_acc'])<1e-7
   audit['checks'].append({'suite':suite,'regime':regime,'seed':seed,'alignment_direct_sum_agrees':True,'checkpoint_id_accuracy_matches':True,'ablation_identity':True,'full_patch_identity':True,'noop_identity':True,'frozen_bitwise_unchanged':True if frozen else None})
   detail={'suite':suite,'regime':regime,'seed':seed,'dataset_sha256':hashes,'class_counts':{k:np.bincount(dset.y,minlength=4).tolist() for k,dset in ds.items()},'confusion':confusions,'patching':probes,'noop':{'positive_shift':0.0,'mean_shift':0.0,'prediction_changed':0.0,'source_prediction':float((L[tt].argmax(1)==Y[ss]).float().mean())},'topk_repo_ks':met['topk_ks'],'topk_repo_success':r.patchk_success.tolist(),'topk_repo_shift':r.patchk_shift.tolist()}
   detailed.append(detail);rows.append(row)
   cachepath(path).write_text(json.dumps({'row':row,'detail':detail,'check':audit['checks'][-1],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},indent=2))
   print('ANALYZED',suite,regime,seed,flush=True)
cached=[json.loads(p.read_text()) for p in sorted(cache.glob('*.json'))]
rows=[x['row'] for x in cached];detailed=[x['detail'] for x in cached];audit['checks']=[x['check'] for x in cached]
pd.DataFrame(rows).to_csv(report/'per_run.csv',index=False)
(report/'details.json').write_text(json.dumps(detailed,indent=2))
(report/'audit.json').write_text(json.dumps(audit,indent=2))
pd.DataFrame(rows).groupby(['suite','regime']).agg(['mean','std']).drop(columns='seed').to_csv(report/'summary.csv')
print('DONE',len(rows))
