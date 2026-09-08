import os,sys,subprocess,time,json
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLBACKEND':'Agg','PYTHONHASHSEED':'0'}
for seed in range(5):
 for control in ['random_unitnorm','frozen_random_unitnorm']:
  out=f'runs/controls/{control}/seed{seed:02d}'
  if Path(out,'control.json').exists():continue
  cmd=[sys.executable,'run_control.py',control,'--device','cpu','--seed',str(seed),'--epochs','100','--train_size','200','--val_size','400','--test_size','600','--num_pairs','1200','--outdir',out]
  t=time.time();print('START',control,seed,flush=True)
  with open(f'logs/control_{control}_{seed}.log','w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=f)
  with open('execution.jsonl','a') as f:f.write(json.dumps(dict(suite='controls',regime=control,seed=seed,command=cmd,seconds=time.time()-t,returncode=r.returncode))+'\n')
  print('END',control,seed,round(time.time()-t,2),r.returncode,flush=True)
  if r.returncode:sys.exit(r.returncode)
