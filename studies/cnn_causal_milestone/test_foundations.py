import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from study import *
torch.set_num_threads(2)
T=bank();flat=T.reshape(16,-1).astype(float)
assert np.allclose(flat.mean(1),0,atol=1e-7)
assert np.allclose(np.linalg.norm(flat,axis=1),1,atol=1e-6)
cos=np.sum(flat[8:12,None,:]*flat[None,:8,:],axis=-1)
assert np.max(np.abs(cos))<.95
banks=[]
for b in range(10):
 S=spectrum_bank(T,b);f=S.reshape(16,-1).astype(float)
 assert np.allclose(flat@flat.T,f@f.T,atol=2e-6)
 assert np.allclose(np.abs(np.fft.fft2(T)),np.abs(np.fft.fft2(S)),atol=2e-6)
 assert np.allclose(np.linalg.svd(flat,compute_uv=False),np.linalg.svd(f,compute_uv=False),atol=2e-6)
 banks.append({'block':b,'alignment':alignment(S[:,None],T),'gram_max_error':float(np.abs(flat@flat.T-f@f.T).max()),'power_max_error':float(np.abs(np.abs(np.fft.fft2(T))**2-np.abs(np.fft.fft2(S))**2).max())})
streams=[seed(10,b,t,s) for b in range(10) for t in range(2) for s in range(3)]
assert len(streams)==len(set(streams))
a=make_data('two_concepts',99999,0,2);b=make_data('two_concepts',99999,1,2);a2=make_data('two_concepts',99999,0,2)
assert np.array_equal(a['x'],a2['x']) and not np.array_equal(a['x'],b['x'])
assert np.array_equal(np.bincount(a['y']),np.array([2]*4))
bas,cf,c=pairs('two_concepts',2)
assert np.all((a['concepts'][bas]!=a['concepts'][cf]).sum(1)==1)
for ar in ARCHS:
 tails=[]
 for cond in CONDITIONS:
  m,anchor,lam=setup(ar,cond,99999,'single_shape');tails.append({k:v for k,v in m.state_dict().items() if k!='conv.weight'})
  if lam:
   w=m.conv.weight
   with torch.no_grad():w.add_(.01)
   loss=lam*(w-anchor).square().sum()/anchor.square().sum();loss.backward()
   assert torch.allclose(w.grad,2*lam*(w-anchor)/anchor.square().sum())
 assert all(all(torch.equal(tails[0][k],d[k]) for k in tails[0]) for d in tails)
 # Nondivisible batch denominator check against direct forward.
 m.eval();l,_,_,_=collect(m,a['x']);assert np.isclose((l.argmax(1).numpy()==a['y']).mean(),float((m(torch.from_numpy(a['x'])).argmax(1)==torch.from_numpy(a['y'])).float().mean()))
audit={'template_rank_tol_1e-6':int((np.linalg.svd(flat,compute_uv=False)>1e-6).sum()),'singular_values':np.linalg.svd(flat,compute_uv=False).tolist(),'max_corner_edge_abs_cosine':float(np.abs(cos).max()),'spectrum_controls':banks,'stream_ids_unique':True,'factorial_pairs_change_one_concept':True,'deterministic_renderer':True,'head_initialization_matched':True,'retention_gradient_verified':True,'sample_weighting_verified':True}
(ROOT/'foundation_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
