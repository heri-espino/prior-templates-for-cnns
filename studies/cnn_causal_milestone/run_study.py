import argparse,os,time,json,hashlib,platform
from datetime import datetime,timezone
from study import *
ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=1);args=ap.parse_args()
torch.set_num_threads(2)
for block in range(args.worker,10,args.workers):
 for task in TASKS:
  data=data_for(task,block)
  for arch in ARCHS:
   # Same order in every block; outcomes do not alter execution or settings.
   for condition in CONDITIONS:
    out=ROOT/'runs'/task/arch/f'block{block:02d}'/condition
    if (out/'result.json').exists():continue
    print('START',block,task,arch,condition,flush=True);start=time.time()
    try:
     run_one(task,arch,block,condition,data)
     record=dict(block=block,task=task,architecture=arch,condition=condition,seconds=time.time()-start,status='complete',utc=datetime.now(timezone.utc).isoformat())
     with open(ROOT/f'execution_worker{args.worker}.jsonl','a') as f:f.write(json.dumps(record)+'\n')
     print('DONE',block,task,arch,condition,round(record['seconds'],2),flush=True)
    except Exception as e:
     with open(ROOT/f'execution_worker{args.worker}.jsonl','a') as f:f.write(json.dumps(dict(block=block,task=task,architecture=arch,condition=condition,status='failed',error=repr(e)))+'\n')
     raise
