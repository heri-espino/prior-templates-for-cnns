"""Run repaired repository entry point without changing experimental defaults."""
import os,sys,time,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
os.chdir(root)
logdir=root/'logs'; logdir.mkdir(exist_ok=True)
env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','MPLBACKEND':'Agg','PYTHONHASHSEED':'0'}
for suite,seeds,n,v,t,epochs,pairs in [('v4_lowdata',range(5),200,400,600,100,1200),('v3_seed0',[0],8000,1000,1000,10,1500)]:
 for seed in seeds:
  for regime in ['random','template_init','frozen_templates']:
   out=f'runs/{suite}/{regime}/seed{seed:02d}'
   if Path(out,'model.pt').exists(): continue
   cmd=[sys.executable,'-m','src.train','--device','cpu','--regime',regime,'--seed',str(seed),'--epochs',str(epochs),'--train_size',str(n),'--val_size',str(v),'--test_size',str(t),'--run_probes','1','--num_pairs',str(pairs),'--outdir',out]
   started=time.time()
   print('START',suite,regime,seed,flush=True)
   with open(logdir/f'{suite}_{regime}_{seed}.log','w') as f:
    r=subprocess.run(cmd,env=env,stdout=f,stderr=f)
   with open('execution.jsonl','a') as f:f.write(json.dumps(dict(suite=suite,regime=regime,seed=seed,command=cmd,seconds=time.time()-started,returncode=r.returncode))+'\n')
   print('END',suite,regime,seed,'seconds',round(time.time()-started,2),'returncode',r.returncode,flush=True)
   if r.returncode: sys.exit(r.returncode)
