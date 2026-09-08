from pathlib import Path
import json,numpy as np,torch
from src.models.cnn import TinyCNN,ModelConfig
from src.data.shapes import ShapesDataset,default_splits
from src.templates.primitives import TemplateSpec,make_template_bank,template_names
from src.utils.seed import set_seed
T=make_template_bank(TemplateSpec()); names=template_names(TemplateSpec());A=T.reshape(16,-1)@T.reshape(16,-1).T
r={'corner_matches':[],'paired_initialization_checks':[]}
for i in range(8,12):
 j=int(np.abs(A[i,:8]).argmax());r['corner_matches'].append({'corner':names[i],'edge':names[j],'signed_cosine':float(A[i,j])})
for seed in range(5):
 heads=[];orders=[]
 for reg in ['random','template_init','frozen_templates']:
  set_seed(seed);m=TinyCNN(ModelConfig(regime=reg),T if reg!='random' else None)
  heads.append(torch.cat([m.classifier.weight.flatten(),m.classifier.bias]))
  orders.append(torch.randperm(200))
 assert all(torch.equal(heads[0],x) for x in heads)
 assert all(torch.equal(orders[0],x) for x in orders)
 r['paired_initialization_checks'].append({'seed':seed,'heads_identical':True,'post_construction_rng_identical':True})
a=ShapesDataset(default_splits(0,200,400,600)['val'])
b=ShapesDataset(default_splits(1,200,400,600)['train'])
assert np.array_equal(a.x[:200],b.x) and np.array_equal(a.y[:200],b.y)
r['cross_seed_overlap']={'seed0_val_prefix_equals_seed1_train':True,'exact_images':200}
Path('analysis').mkdir(exist_ok=True);Path('analysis/design_checks.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
