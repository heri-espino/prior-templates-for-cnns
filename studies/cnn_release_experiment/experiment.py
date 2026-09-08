"""Fresh-block retention-release experiment, with atomic epoch resume."""
import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import core

CONDITIONS=['random','random_unitnorm','template_init','template_retention_1','template_release']

def strength(condition, epoch, release_start, release_end):
    if condition=='template_retention_1': return 1.
    if condition!='template_release': return 0.
    return float(np.clip((release_end-epoch)/(release_end-release_start),0,1))

def atomic_json(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)

def atomic_checkpoint(path, value):
    tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)

def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    core.ROOT=root
    hashes={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['core.py','experiment.py']}
    config={k:v for k,v in vars(args).items() if k not in ['threads','output']}
    config.update(source_hashes=hashes,master_seed=core.MASTER,lr=.003,batch_size=128)
    manifest=root/'design.json'
    if manifest.exists():
        if json.loads(manifest.read_text())!=config: raise ValueError('Settings/source differ from saved design. Choose a new --output directory.')
    else: atomic_json(manifest,config)
    atomic_json(root/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),threads=args.threads))
    for block in range(args.start_block,args.start_block+args.blocks):
      for task in args.tasks:
        data=core.data_for(task,block)
        for arch in args.architectures:
          for condition in args.conditions:
            out=root/'runs'/task/arch/f'block{block:04d}'/condition;out.mkdir(parents=True,exist_ok=True)
            if (out/'complete.json').exists(): continue
            print(f'START {task} {arch} block={block} {condition}',flush=True)
            m,anchor,_=core.setup(arch,'template_init' if condition=='template_release' else condition,block,task)
            init=m.conv.weight.detach().clone()
            opt=torch.optim.Adam(m.parameters(),lr=.003)
            gen=torch.Generator().manual_seed(core.seed(21,block,core.TASKS.index(task)))
            history=[];first=1;resume=out/'latest.pt'
            if resume.exists():
                saved=torch.load(resume,map_location='cpu',weights_only=False)
                m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);gen.set_state(saved['shuffle_rng']);torch.set_rng_state(saved['torch_rng'])
                history=saved['history'];first=saved['epoch']+1
                if saved['epoch'] in args.checkpoints or saved['epoch']==args.epochs:
                    atomic_checkpoint(out/f"epoch_{saved['epoch']:04d}.pt",saved)
            else:
                np.save(out/'conv_initial.npy',init.numpy())
                atomic_checkpoint(out/'epoch_0000.pt',dict(epoch=0,model=m.state_dict()))
            atomic_json(out/'config.json',dict(task=task,architecture=arch,block=block,condition=condition,data_hashes={k:core.digest(v['x']) for k,v in data.items()},**config))
            X=torch.from_numpy(data['train']['x']);Y=torch.from_numpy(data['train']['y'])
            # Repair missing evaluations after interruption before continuing training.
            for ep in range(1,first):
                cp=out/f'epoch_{ep:04d}.pt';ev=out/'evaluations'/f'epoch_{ep:04d}'
                if cp.exists() and not (ev/'result.json').exists():
                    probe=core.Network(arch);probe.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
                    core.evaluate(probe,task,arch,block,condition,data,ev,init)
            for epoch in range(first,args.epochs+1):
                start=time.perf_counter();m.train();order=torch.randperm(len(X),generator=gen)
                ce_sum=reg_sum=correct=grad_sum=0.;batches=0
                lam=strength(condition,epoch,args.release_start,args.release_end)
                for st in range(0,len(X),128):
                    ix=order[st:st+128];opt.zero_grad(set_to_none=True);logits=m(X[ix]);ce=F.cross_entropy(logits,Y[ix])
                    reg=(m.conv.weight-anchor).square().sum()/anchor.square().sum() if anchor is not None else torch.tensor(0.)
                    (ce+lam*reg).backward();grad_sum+=float(m.conv.weight.grad.norm());batches+=1;opt.step()
                    ce_sum+=float(ce.detach())*len(ix);reg_sum+=float(reg.detach())*len(ix);correct+=int((logits.detach().argmax(1)==Y[ix]).sum())
                train_seconds=time.perf_counter()-start
                m.eval();vl,vz,_,_=core.collect(m,data['val']['x'])
                history.append(dict(epoch=epoch,optimizer_steps=epoch*4,lambda_retention=lam,train_ce=ce_sum/len(X),train_acc=correct/len(X),retention_penalty=reg_sum/len(X),val_ce=float(F.cross_entropy(vl,torch.from_numpy(data['val']['y']))),val_acc=float((vl.argmax(1).numpy()==data['val']['y']).mean()),alignment=core.alignment(m.conv.weight.detach().numpy(),core.bank()),kernel_norm=float(m.conv.weight.detach().norm()),conv_gradient_norm=grad_sum/batches,val_pooled_feature_norm=float(vz.norm(dim=1).mean()),train_seconds=train_seconds))
                state=dict(epoch=epoch,model=m.state_dict(),optimizer=opt.state_dict(),shuffle_rng=gen.get_state(),torch_rng=torch.get_rng_state(),history=history)
                atomic_checkpoint(resume,state);atomic_json(out/'history.json',history)
                if epoch in args.checkpoints or epoch==args.epochs:
                    atomic_checkpoint(out/f'epoch_{epoch:04d}.pt',state)
                    core.evaluate(m,task,arch,block,condition,data,out/'evaluations'/f'epoch_{epoch:04d}',init)
                    print(f'  epoch {epoch}/{args.epochs}: val={history[-1]["val_acc"]:.3f}, alignment={history[-1]["alignment"]:.3f}, lambda={lam:.3f}',flush=True)
            # Resume files retain the authoritative history even if JSON writing was interrupted.
            atomic_json(out/'history.json',history)
            atomic_json(out/'complete.json',dict(epochs=args.epochs,status='complete'))
    from summarize import summarize
    summarize(root)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/main');p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--blocks',type=int,default=10);p.add_argument('--start-block',type=int,default=2000)
    p.add_argument('--tasks',nargs='+',choices=core.TASKS,default=core.TASKS)
    p.add_argument('--architectures',nargs='+',choices=core.ARCHS,default=core.ARCHS)
    p.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=CONDITIONS)
    p.add_argument('--release-start',type=int,default=10);p.add_argument('--release-end',type=int,default=80)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[1,5,10,20,40,80,120,160,200])
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    if min(a.epochs,a.blocks,a.threads)<1 or a.start_block<2000 or not 0<=a.release_start<a.release_end: p.error('Positive epochs/blocks/threads, fresh start-block >=2000, and 0 <= release-start < release-end required.')
    run(a)
