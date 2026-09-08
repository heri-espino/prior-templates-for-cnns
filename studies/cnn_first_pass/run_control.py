"""Added controls; original src model is unchanged. Forward/trainer are reused."""
import sys,json
from pathlib import Path
import torch
import src.train as train
from src.models.cnn import TinyCNN
control=sys.argv[1]; del sys.argv[1]
assert control in ('random_unitnorm','frozen_random_unitnorm')
class ControlCNN(TinyCNN):
 def __init__(self,cfg,templates=None):
  assert cfg.regime=='random'
  super().__init__(cfg,templates)
  with torch.no_grad():
   w=self.conv.weight
   w.sub_(w.mean((1,2,3),keepdim=True))
   w.div_(torch.linalg.vector_norm(w,dim=(1,2,3),keepdim=True))
  if control=='frozen_random_unitnorm':
   self.conv.weight.register_hook(lambda grad: grad*0)
train.TinyCNN=ControlCNN
train.main()
out=Path(sys.argv[sys.argv.index('--outdir')+1])
(out/'control.json').write_text(json.dumps({'control':control,'base_regime':'random','description':'Center each random kernel and normalize to unit L2; optionally zero all convolution gradients. Classifier initialization is unchanged.'},indent=2))
