from __future__ import annotations
import os,sys,json,hashlib,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image,ImageDraw
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent
TASKS=['single_shape','two_concepts'];ARCHS=['TinyCNN','TwoLayerCNN']
CONDITIONS=['random','random_unitnorm','frozen_random_unitnorm','template_init','template_retention_0.1','template_retention_1','frozen_templates','spectrum_init','spectrum_retention_1','frozen_spectrum']
MASTER=2026090617
NAMES=[f'edge_{i:02d}' for i in range(8)]+[f'corner_{i:02d}' for i in range(4)]+[f'ring_{i:02d}' for i in range(4)]
def seed(*keys):return int(np.random.SeedSequence([MASTER,*map(int,keys)]).generate_state(1)[0])
def rng(*keys):return np.random.default_rng(np.random.SeedSequence([MASTER,*map(int,keys)]))
def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def normalize(x):
 x=np.asarray(x,dtype=np.float64);x=x-x.mean(axis=(-2,-1),keepdims=True);return x/np.sqrt((x*x).sum(axis=(-2,-1),keepdims=True))
def bank():
 ax=np.arange(-4,5);x,y=np.meshgrid(ax,ax);g=np.exp(-(x*x+y*y)/8);out=[]
 for i in range(8):
  a=i*np.pi/8;out.append((x*np.cos(a)+y*np.sin(a))*g)
 for i in range(4):
  a=i*np.pi/2;u=x*np.cos(a)+y*np.sin(a);v=-x*np.sin(a)+y*np.cos(a)
  d1=v*v+np.minimum(u,0)**2;d2=u*u+np.minimum(v,0)**2
  out.append(np.exp(-np.minimum(d1,d2)/(2*.65**2))*g)
 rad=np.sqrt(x*x+y*y)
 for r in [1.5,2.,2.5,3.]:out.append(np.exp(-(rad-r)**2/2)-np.exp(-(rad-r-1)**2/2))
 return normalize(np.stack(out)).astype(np.float32)
def spectrum_bank(T,block):
 field=rng(30,block).normal(size=(9,9));H=np.fft.fft2(field);phase=H/np.abs(H)
 return np.fft.ifft2(np.fft.fft2(T,axes=(-2,-1))*phase,axes=(-2,-1)).real.astype(np.float32)
def alignment(W,T):
 a=normalize(W[:,0]).reshape(16,-1);b=normalize(T).reshape(16,-1)
 A=(a[:,None,:]*b[None,:,:]).sum(-1)
 return float(A.max(1).mean())
def render_object(shape,cx,cy,radius,theta,width,aa=3):
 im=Image.new('L',(32*aa,32*aa),0);d=ImageDraw.Draw(im);cx*=aa;cy*=aa;radius*=aa;w=max(1,round(width*aa))
 if shape=='line':
  dx=radius*np.cos(theta);dy=radius*np.sin(theta);d.line([(cx-dx,cy-dy),(cx+dx,cy+dy)],fill=255,width=w)
 elif shape=='circle':d.ellipse([cx-radius,cy-radius,cx+radius,cy+radius],outline=255,width=w)
 else:
  n=3 if shape=='triangle' else 4
  pts=[(cx+radius*np.cos(theta+j*2*np.pi/n),cy+radius*np.sin(theta+j*2*np.pi/n)) for j in range(n)]
  d.line(pts+[pts[0]],fill=255,width=w,joint='curve')
 return np.asarray(im.resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)/255

def make_data(task,block,split,nctx):
 ti=TASKS.index(task);ri=rng(10,block,ti,split);x=[];masks=[];meta=[];contexts=[]
 for j in range(nctx):
  theta=float(ri.uniform(0,np.pi/4));width=float(ri.uniform(.8,1.6));scale=float(ri.uniform(.85,1.15));tx,ty=ri.uniform(-2,2,2);noise_sd=float(ri.uniform(0,.05));noise=ri.normal(0,noise_sd,(32,32)).astype(np.float32)
  occ=bool(ri.random()<.1);ox,oy=ri.integers(0,27,2);swapped=bool(ri.integers(2));nu_seed=seed(11,block,ti,split,j)
  # A second nuisance realization uses a separate per-context namespace.
  nr=np.random.default_rng(nu_seed);ntx,nty=nr.uniform(-2,2,2);nnoise=nr.normal(0,noise_sd,(32,32)).astype(np.float32)
  context=dict(theta=theta,width=width,scale=scale,tx=float(tx),ty=float(ty),noise_sd=noise_sd,noise_sha256=digest(noise),occluded=occ,ox=int(ox),oy=int(oy),swapped=swapped,nuisance_seed=nu_seed,ntx=float(ntx),nty=float(nty),nnoise_sha256=digest(nnoise))
  states=[];states_n=[];state_masks=[]
  for state in range(4):
   def draw(dx,dy):
    if task=='single_shape':
     shape=['line','circle','triangle','square'][state];arr=render_object(shape,16+dx,16+dy,10*scale,theta,width);ms=np.stack([arr if c==state else np.zeros_like(arr) for c in range(4)])
    else:
     loc=[(8+dx/2,15+dy),(24+dx/2,17+dy)]
     if swapped:loc=loc[::-1]
     a=render_object('circle' if state&1 else 'square',*loc[0],4.5*scale,theta,width)
     b=render_object('triangle' if state&2 else 'line',*loc[1],4.5*scale,theta,width)
     arr=np.maximum(a,b);ms=np.stack([a if state&1 else np.zeros_like(a),b if state&2 else np.zeros_like(b)])
    if occ:arr[oy:oy+5,ox:ox+5]=0;ms[:,oy:oy+5,ox:ox+5]=0
    return arr,ms
   arr,ms=draw(tx,ty);arr_n,_=draw(ntx,nty)
   states.append(np.clip(arr+noise,0,1));states_n.append(np.clip(arr_n+nnoise,0,1));state_masks.append(ms>.2)
  x.extend(states);masks.extend(state_masks);contexts.extend(states_n);meta.append(context)
 X=np.asarray(x,dtype=np.float32)[:,None];Y=np.tile(np.arange(4),nctx).astype(np.int64);M=np.asarray(masks,dtype=bool)
 C=np.eye(4,dtype=np.int64)[Y] if task=='single_shape' else np.stack([Y&1,(Y>>1)&1],1)
 return dict(x=X,y=Y,concepts=C,masks=M,nuisance_x=np.asarray(contexts,dtype=np.float32)[:,None],meta=meta,stream_seed=seed(10,block,ti,split))
def data_for(task,block):
 d=ROOT/'data'/task/f'block{block:02d}';d.mkdir(parents=True,exist_ok=True);out={}
 for split,n in [('train',128),('val',64),('test',64)]:
  p=d/f'{split}.npz';sp=['train','val','test'].index(split)
  if not p.exists():
   a=make_data(task,block,sp,n);np.savez_compressed(p,**{k:v for k,v in a.items() if k not in ['meta','stream_seed']})
   (d/f'{split}.json').write_text(json.dumps({'stream_seed':a['stream_seed'],'image_sha256':digest(a['x']),'nuisance_contexts':a['meta']},indent=2))
  with np.load(p) as z:out[split]={k:z[k] for k in z.files}
 return out
class Network(nn.Module):
 def __init__(self,arch):
  super().__init__();self.arch=arch;self.conv=nn.Conv2d(1,16,9,padding=4,bias=False)
  self.conv2=nn.Conv2d(16,16,3,padding=1,bias=False) if arch=='TwoLayerCNN' else None
  self.classifier=nn.Linear(16,4)
 def features(self,x):return F.relu(self.conv(x))
 def tail_features(self,h):
  if self.conv2 is not None:h=F.relu(self.conv2(h))
  return h.amax((2,3))
 def tail(self,h):return self.classifier(self.tail_features(h))
 def forward(self,x):return self.tail(self.features(x))
def setup(arch,condition,block,task):
 torch.manual_seed(seed(20,block,TASKS.index(task),ARCHS.index(arch)))
 m=Network(arch);T=bank();S=spectrum_bank(T,block);anchor=None;lam=0.
 with torch.no_grad():
  if condition in ['random_unitnorm','frozen_random_unitnorm']:
   w=m.conv.weight;w.sub_(w.mean((1,2,3),keepdim=True));w.div_(torch.linalg.vector_norm(w,dim=(1,2,3),keepdim=True))
  if condition.startswith('template') or condition=='frozen_templates':anchor=torch.from_numpy(T[:,None].copy());m.conv.weight.copy_(anchor)
  if 'spectrum' in condition:anchor=torch.from_numpy(S[:,None].copy());m.conv.weight.copy_(anchor)
 if 'retention' in condition:lam=float(condition.split('_')[-1])
 if condition.startswith('frozen'):m.conv.weight.requires_grad_(False)
 return m,anchor,lam
@torch.no_grad()
def collect(m,X,maps=False):
 L=[];Z=[];H=[];P=[]
 for i in range(0,len(X),128):
  h=m.features(torch.from_numpy(X[i:i+128]));p=m.tail_features(h);L.append(m.classifier(p));Z.append(h.amax((2,3)));P.append(p)
  if maps:H.append(h)
 return torch.cat(L),torch.cat(Z),torch.cat(P),torch.cat(H) if maps else None

def dot(a,b):return np.einsum('ij,jk->ik',a,b,optimize=False)
def fit_head(P,Y,V,Q,QY):
 mu=P.mean(0);sd=P.std(0);sd[sd<1e-6]=1.;X=np.column_stack([(P-mu)/sd,np.ones(len(P))]).astype(float);XV=np.column_stack([(V-mu)/sd,np.ones(len(V))]);XT=np.column_stack([(Q-mu)/sd,np.ones(len(Q))]);K=4
 fits=[]
 for alpha in [0.,.001,.1]:
  def fg(w):
   w=w.reshape(17,K);l=dot(X,w);lp=l-logsumexp(l,axis=1,keepdims=True);prob=np.exp(lp);prob[np.arange(len(Y)),Y]-=1
   penalty=w.copy();penalty[-1]=0
   val=-lp[np.arange(len(Y)),Y].mean()+alpha*.5*(penalty*penalty).sum();grad=dot(X.T,prob)/len(Y)+alpha*penalty
   return val,grad.ravel()
  res=minimize(fg,np.zeros(17*K),jac=True,method='L-BFGS-B',options={'maxiter':500,'gtol':1e-7,'ftol':1e-12})
  W=res.x.reshape(17,K);lv=dot(XV,W);lv=lv-logsumexp(lv,axis=1,keepdims=True)
  fits.append(dict(alpha=alpha,success=bool(res.success),message=str(res.message),iterations=int(res.nit),gradient_max=float(np.max(np.abs(res.jac))),train_acc=float((dot(X,W).argmax(1)==Y).mean()),val_ce=float(-lv[np.arange(len(QY['val'])),QY['val']].mean()),val_acc=float((lv.argmax(1)==QY['val']).mean()),test_acc=float((dot(XT,W).argmax(1)==QY['test']).mean()),weights=W.tolist(),mean=mu.tolist(),sd=sd.tolist()))
 best=min(range(len(fits)),key=lambda i:fits[i]['val_ce']);return dict(selected=best,selected_fit=fits[best],all_fits=fits)
def auc(y,score):
 y=np.asarray(y,dtype=bool);n=int(y.sum());m=len(y)-n
 if not n or not m:return float('nan')
 r=rankdata(score);return float((r[y].sum()-n*(n+1)/2)/(n*m))
def concept_orders(Z,C):
 z=Z.numpy();sd=z.std(0)+1e-6;effects=np.stack([(z[C[:,i]==1].mean(0)-z[C[:,i]==0].mean(0))/sd for i in range(C.shape[1])])
 return np.argsort(-np.abs(effects),axis=1),np.where(effects>=0,1.,-1.),effects

def pairs(task,nctx):
 b=[];c=[];concept=[]
 for j in range(nctx):
  for s in range(4):
   targets=[t for t in range(4) if t!=s] if task=='single_shape' else [s^1,s^2]
   for t in targets:b.append(j*4+s);c.append(j*4+t);concept.append(t if task=='single_shape' else (0 if (s^t)==1 else 1))
 return np.array(b),np.array(c),np.array(concept)
@torch.no_grad()
def patch_eval(m,H,L,orders,task,random_orders):
 b,c,g=pairs(task,len(H)//4);p0=L[b].softmax(1);p1=L[c].softmax(1);y1=torch.tensor(c%4);den=float(((p1-p0)**2).sum());baseline=L[b].argmax(1);actual=L[c].argmax(1);good=(baseline==torch.tensor(b%4))&(actual==y1)
 def measure(prob):
  err=float(((prob-p1)**2).sum());pred=prob.argmax(1)
  return dict(fidelity=1-err/den if den>=1e-10 else None,cf_accuracy=float((pred==y1).float().mean()),agreement=float((pred==actual).float().mean()),cf_accuracy_both_correct=float((pred[good]==y1[good]).float().mean()) if good.any() else None)
 def run(o,k):
  ps=[]
  for st in range(0,len(b),128):
   bs=b[st:st+128];cs=c[st:st+128];gs=g[st:st+128];h=H[bs].clone();ix=torch.from_numpy(o[gs,:k].copy());bi=torch.arange(len(bs))[:,None]
   h[bi,ix]=H[cs][bi,ix];ps.append(m.tail(h).softmax(1))
  return torch.cat(ps)
 out={'denominator':den,'pair_count':len(b),'both_correct_count':int(good.sum()),'noop':measure(p0),'full':measure(p1),'selected':{},'random':[]}
 for k in [1,4,8]:out['selected'][str(k)]=measure(run(orders,k))
 for ro in random_orders:out['random'].append(measure(run(ro,4)))
 full=run(orders,16);assert torch.allclose(full,p1,atol=2e-6,rtol=1e-5)
 assert abs(out['noop']['fidelity'])<1e-6 if den>=1e-10 else out['noop']['fidelity'] is None
 return out

def run_one(task,arch,block,condition,data,epochs=40,output_root='runs'):
 out=ROOT/output_root/task/arch/f'block{block:02d}'/condition;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
 start=time.time();m,anchor,lam=setup(arch,condition,block,task);init=m.conv.weight.detach().clone();T=bank()
 config=dict(task=task,architecture=arch,block=block,condition=condition,epochs=epochs,lr=.003,batch_size=128,lambda_retention=lam,image_size=32,channels=16,kernel_size=9,model_seed=seed(20,block,TASKS.index(task),ARCHS.index(arch)),shuffle_seed=seed(21,block,TASKS.index(task)),data_hashes={k:digest(v['x']) for k,v in data.items()})
 (out/'config.json').write_text(json.dumps(config,indent=2));opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.003);gen=torch.Generator().manual_seed(config['shuffle_seed'])
 X=torch.from_numpy(data['train']['x']);Y=torch.from_numpy(data['train']['y']);history=[]
 for ep in range(epochs):
  m.train();order=torch.randperm(len(X),generator=gen);ls=0.;rs=0.;correct=0
  for st in range(0,len(X),128):
   ix=order[st:st+128];opt.zero_grad(set_to_none=True);logits=m(X[ix]);ce=F.cross_entropy(logits,Y[ix]);reg=((m.conv.weight-anchor)**2).sum()/anchor.square().sum() if lam else torch.tensor(0.)
   loss=ce+lam*reg;loss.backward();opt.step();ls+=float(ce.detach())*len(ix);rs+=float(reg.detach())*len(ix);correct+=int((logits.detach().argmax(1)==Y[ix]).sum())
  m.eval();vl,_,_,_=collect(m,data['val']['x']);history.append(dict(epoch=ep+1,train_ce=ls/len(X),retention_penalty=rs/len(X),train_acc=correct/len(X),val_ce=float(F.cross_entropy(vl,torch.from_numpy(data['val']['y']))),val_acc=float((vl.argmax(1).numpy()==data['val']['y']).mean())))
 torch.save(m.state_dict(),out/'model.pt');np.save(out/'conv_initial.npy',init.numpy());np.save(out/'conv_final.npy',m.conv.weight.detach().numpy());(out/'history.json').write_text(json.dumps(history,indent=2))
 m.eval();lt,zt,pt,_=collect(m,data['train']['x']);lv,zv,pv,hv=collect(m,data['val']['x'],True);le,ze,pe,he=collect(m,data['test']['x'],True)
 orders,signs,effects=concept_orders(zv,data['val']['concepts']);sem=[];ious=[];thresholds=[]
 loc_channels=[]
 for c in range(orders.shape[0]):
  i=orders[c,0];sem.append(auc(data['test']['concepts'][:,c],ze[:,i].numpy()*signs[c,i]))
  vp=data['val']['concepts'][:,c]==1;vm=data['val']['masks'][vp,c]
  thrs=np.quantile(hv.numpy(),.95,axis=(0,2,3));loc_scores=[]
  for ch in range(16):
   prediction=hv[vp,ch].numpy()>thrs[ch];un=np.logical_or(prediction,vm).sum();loc_scores.append(np.logical_and(prediction,vm).sum()/un if un else 0.)
  i=int(np.argmax(loc_scores));thr=float(thrs[i]);thresholds.append(thr);loc_channels.append(i)
  positive=data['test']['concepts'][:,c]==1;pred=he[positive,i].numpy()>thr;truth=data['test']['masks'][positive,c];union=np.logical_or(pred,truth).sum();ious.append(float(np.logical_and(pred,truth).sum()/union) if union else 0.)
 ro_rng=rng(40,block,TASKS.index(task));random_orders=[np.stack([ro_rng.permutation(16) for _ in range(orders.shape[0])]) for _ in range(8)]
 patch=patch_eval(m,he,le,orders,task,random_orders);ln,zn,_,_=collect(m,data['test']['nuisance_x'])
 heads=fit_head(pt.numpy(),data['train']['y'],pv.numpy(),pe.numpy(),{'val':data['val']['y'],'test':data['test']['y']})
 w=m.conv.weight.detach().numpy();frozen=condition.startswith('frozen')
 if frozen:assert np.array_equal(init.numpy(),w)
 scalar=dict(task=task,architecture=arch,block=block,condition=condition,id_acc=float((le.argmax(1).numpy()==data['test']['y']).mean()),test_ce=float(F.cross_entropy(le,torch.from_numpy(data['test']['y']))),alignment=alignment(w,T),initial_alignment=alignment(init.numpy(),T),semantic_auc=float(np.mean(sem)),localization_iou=float(np.mean(ious)),fidelity_selected=patch['selected']['4']['fidelity'],fidelity_random=float(np.mean([r['fidelity'] for r in patch['random']])) if patch['denominator']>=1e-10 else None,cf_accuracy=patch['selected']['4']['cf_accuracy'],cf_accuracy_random=float(np.mean([r['cf_accuracy'] for r in patch['random']])),nuisance_acc=float((ln.argmax(1).numpy()==data['test']['y']).mean()),nuisance_prediction_stability=float((ln.argmax(1)==le.argmax(1)).float().mean()),nuisance_probability_l2=float(((ln.softmax(1)-le.softmax(1))**2).sum(1).mean()),refit_test_acc=heads['selected_fit']['test_acc'],refit_val_acc=heads['selected_fit']['val_acc'],refit_converged=heads['selected_fit']['success'],drift_l2=float(np.linalg.norm((w-init.numpy()).reshape(16,-1),axis=1).mean()),seconds=time.time()-start)
 scalar['causal_usefulness']=scalar['fidelity_selected']-scalar['fidelity_random'] if scalar['fidelity_selected'] is not None else None
 np.savez_compressed(out/'probe_arrays.npz',concept_orders=orders,concept_effects=effects,semantic_auc=np.array(sem),localization_iou=np.array(ious),thresholds=np.array(thresholds),localization_channels=np.array(loc_channels),test_logits=le.numpy(),test_z=ze.numpy(),test_pooled=pe.numpy(),val_z=zv.numpy())
 (out/'patching.json').write_text(json.dumps(patch,indent=2));(out/'head_refit.json').write_text(json.dumps(heads,indent=2));(out/'checks.json').write_text(json.dumps(dict(full_patch_recovers_input_cf=True,noop_fidelity_zero=True,frozen_unchanged=True if frozen else None),indent=2));(out/'result.json').write_text(json.dumps(scalar,indent=2));return scalar
