"""Supplemental first-layer nuisance-response measurements; no training changes."""
import json
import numpy as np
import pandas as pd
import torch
from study import *
torch.set_num_threads(1);out=ROOT/'analysis/nuisance_cache';out.mkdir(parents=True,exist_ok=True)
for p in sorted((ROOT/'runs').glob('*/*/block*/*/result.json')):
 r=json.loads(p.read_text());key=f"{r['task']}_{r['architecture']}_{r['block']}_{r['condition']}.json";dest=out/key
 if dest.exists():continue
 m=Network(r['architecture']);m.load_state_dict(torch.load(p.parent/'model.pt',map_location='cpu',weights_only=True));m.eval()
 with np.load(ROOT/'data'/r['task']/f"block{r['block']:02d}"/'test.npz') as z:x=z['x'];xn=z['nuisance_x']
 _,z,_,_=collect(m,x);_,zn,_,_=collect(m,xn);bas,cf,g=pairs(r['task'],len(x)//4)
 dz=(z[cf]-z[bas]).square().mean().item();dn=(zn-z).square().mean().item()
 result={k:r[k] for k in ['task','architecture','block','condition']};result.update(concept_z_mean_square=dz,nuisance_z_mean_square=dn,concept_to_nuisance_response_ratio=dz/max(dn,1e-10));dest.write_text(json.dumps(result))
rows=[json.loads(p.read_text()) for p in out.glob('*.json')];pd.DataFrame(rows).to_csv(ROOT/'analysis/nuisance_features.csv',index=False);print('Nuisance feature diagnostics',len(rows))
