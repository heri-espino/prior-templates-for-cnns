import torch,time,json
from study import *
torch.set_num_threads(2)
d=data_for('single_shape',100000)
for ar in ARCHS:
 t=time.time();r=run_one('single_shape',ar,100000,'template_retention_1',d,epochs=2,output_root='smoke_runs');print(json.dumps({'architecture':ar,'seconds':time.time()-t,'execution_complete':True}),flush=True)
