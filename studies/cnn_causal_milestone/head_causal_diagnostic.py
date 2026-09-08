import json
import numpy as np
import pandas as pd
import torch
from study import *
torch.set_num_threads(1);dest=ROOT/'analysis/head_causal_cache';dest.mkdir(parents=True,exist_ok=True)
conditions=['template_init','template_retention_1','frozen_templates','frozen_spectrum','frozen_random_unitnorm']
for p in sorted((ROOT/'runs').glob('*/*/block*/*/result.json')):
 r=json.loads(p.read_text())
 if r['condition'] not in conditions:continue
 q=dest/f"{r['task']}_{r['architecture']}_{r['block']}_{r['condition']}.json"
 if q.exists():continue
 m=Network(r['architecture']);m.load_state_dict(torch.load(p.parent/'model.pt',map_location='cpu',weights_only=True));m.eval()
 fit=json.loads((p.parent/'head_refit.json').read_text())['selected_fit'];W=np.array(fit['weights']);mu=np.array(fit['mean']);sd=np.array(fit['sd']);w=W[:16]/sd[:,None];bias=W[-1]-np.sum(mu[:,None]*w,axis=0)
 with torch.no_grad():m.classifier.weight.copy_(torch.from_numpy(w.T));m.classifier.bias.copy_(torch.from_numpy(bias))
 with np.load(ROOT/'data'/r['task']/f"block{r['block']:02d}"/'test.npz') as z:X=z['x'];Y=z['y']
 L,Z,P,H=collect(m,X,True);acc=float((L.argmax(1).numpy()==Y).mean());assert abs(acc-r['refit_test_acc'])<1e-9
 with np.load(p.parent/'probe_arrays.npz') as z:orders=z['concept_orders']
 rr=rng(40,r['block'],TASKS.index(r['task']));ros=[np.stack([rr.permutation(16) for _ in range(len(orders))]) for _ in range(8)]
 patch=patch_eval(m,H,L,orders,r['task'],ros);random_f=float(np.mean([p['fidelity'] for p in patch['random']]));u=patch['selected']['4']['fidelity']-random_f
 result={k:r[k] for k in ['task','architecture','block','condition']};result.update(original_U=r['causal_usefulness'],refit_U=u,delta_U=u-r['causal_usefulness'],original_cf_accuracy=r['cf_accuracy'],refit_cf_accuracy=patch['selected']['4']['cf_accuracy'],original_acc=r['id_acc'],refit_acc=acc,selected_refit_converged=fit['success'],patching=patch)
 q.write_text(json.dumps(result,indent=2))
rows=[{k:v for k,v in json.loads(p.read_text()).items() if k!='patching'} for p in dest.glob('*.json')];pd.DataFrame(rows).to_csv(ROOT/'analysis/head_causal_diagnostic.csv',index=False);print('Exploratory head-causal evaluations',len(rows))
