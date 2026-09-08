"""Checks scheduling, paired initialization, and exact interrupted resume."""
import argparse,json,tempfile
from pathlib import Path
import torch
import core,experiment

def args(root):
 return argparse.Namespace(output=str(root),epochs=3,blocks=1,start_block=3000,tasks=['single_shape'],architectures=['TinyCNN'],conditions=['template_release'],release_start=1,release_end=3,checkpoints=[1,3],threads=1)

def main():
 assert [experiment.strength('template_release',e,1,3) for e in range(1,5)]==[1,.5,0,0]
 a,_,_=core.setup('TwoLayerCNN','random',3000,'single_shape')
 b,_,_=core.setup('TwoLayerCNN','template_init',3000,'single_shape')
 assert torch.equal(a.classifier.weight,b.classifier.weight) and torch.equal(a.conv2.weight,b.conv2.weight)
 original_eval=core.evaluate
 def stub(m,task,arch,block,condition,data,out,init):
  out.mkdir(parents=True,exist_ok=True);(out/'result.json').write_text(json.dumps(dict(task=task,architecture=arch,block=block,condition=condition)))
 core.evaluate=stub
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp);experiment.run(args(root/'full'))
  original_save=experiment.atomic_json
  def interrupt(path,value):
   if path.name=='history.json' and len(value)==2:raise RuntimeError('simulated interruption after atomic epoch save')
   original_save(path,value)
  experiment.atomic_json=interrupt
  try:experiment.run(args(root/'resume'))
  except RuntimeError as e:assert 'simulated' in str(e)
  else:raise AssertionError('interruption not triggered')
  experiment.atomic_json=original_save;experiment.run(args(root/'resume'))
  sub=Path('runs/single_shape/TinyCNN/block3000/template_release')
  x=torch.load(root/'full'/sub/'latest.pt',weights_only=False)
  y=torch.load(root/'resume'/sub/'latest.pt',weights_only=False)
  assert all(torch.equal(v,y['model'][k]) for k,v in x['model'].items())
  assert all({k:v for k,v in h.items() if k!='train_seconds'}=={k:v for k,v in j.items() if k!='train_seconds'} for h,j in zip(x['history'],y['history']))
 core.evaluate=original_eval
 print('PASS: schedule, paired downstream initialization, exact interrupted training resume')
if __name__=='__main__':main()
