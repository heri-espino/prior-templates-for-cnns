"""Read-only Stage B kernel comparison. No training, inference, or unsafe loading."""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import torch

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / 'studies/cnn_release_experiment/core.py'
METRICS = ['nearest_cosine', 'assignment_cosine', 'matching_gap', 'nearest_template_count',
           'nearest_distance', 'assignment_distance', 'redundancy_cosine', 'mean_raw_norm',
           'initial_index_cosine', 'raw_initial_drift']
BEHAVIOR = ['id_acc', 'semantic_auc', 'localization_iou', 'causal_usefulness', 'cf_accuracy']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(w):
    w = np.asarray(w, dtype=np.float64).reshape(len(w), -1)
    if not np.isfinite(w).all():
        raise ValueError('Nonfinite kernel')
    centered = w - w.mean(1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if (norms < 1e-12).any():
        raise ValueError('Degenerate centered kernel; cosine is undefined')
    return centered / norms


def compare(w, templates, initial):
    a, b, z = normalized(w), normalized(templates), normalized(initial)
    sim = (a[:, None, :] * b[None, :, :]).sum(-1)
    if not np.isfinite(sim).all():
        raise ValueError("Nonfinite pairwise cosine")
    sim = np.clip(sim, -1, 1)
    rows, cols = linear_sum_assignment(-sim)
    nearest = sim.argmax(1)
    best = sim[np.arange(len(w)), nearest]
    assigned = sim[rows, cols]
    mutual = (a[:, None, :] * a[None, :, :]).sum(-1)
    np.fill_diagonal(mutual, -np.inf)
    score = dict(nearest_cosine=best.mean(), assignment_cosine=assigned.mean(),
                 matching_gap=best.mean()-assigned.mean(), nearest_template_count=len(set(nearest)),
                 nearest_distance=np.sqrt(np.maximum(0, 2-2*best)).mean(),
                 assignment_distance=np.sqrt(np.maximum(0, 2-2*assigned)).mean(),
                 redundancy_cosine=mutual.max(1).mean(),
                 mean_raw_norm=np.linalg.norm(w.reshape(len(w), -1), axis=1).mean(),
                 initial_index_cosine=(a*z).sum(1).mean(),
                 raw_initial_drift=np.linalg.norm((w-initial).reshape(len(w), -1), axis=1).mean())
    return score, sim, nearest, cols


def plot_examples(examples, templates, conditions, out):
    allw = [normalized(templates).reshape(16,9,9)]
    allw += [normalized(w).reshape(16,9,9) for w in examples.values()]
    lim = max(float(np.abs(w).max()) for w in allw)
    for task, arch in sorted(set((k[0], k[1]) for k in examples)):
        fig, axes = plt.subplots(1+2*len(conditions),16,figsize=(12,8))
        rows = [('Original bank',templates)]
        for c in conditions:
            for ep in [0,200]:
                w = examples[task,arch,c,ep]
                _,_,_,cols = compare(w,templates,w)
                rows.append((f'{c}\nepoch {ep}',w[np.argsort(cols)]))
        for i,(label,w) in enumerate(rows):
            for j,img in enumerate(normalized(w).reshape(16,9,9)):
                axes[i,j].imshow(img,cmap='RdBu_r',vmin=-lim,vmax=lim)
                axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
                if i == 0: axes[i,j].set_title(f'T{j}',fontsize=10)
            axes[i,0].set_ylabel(label,rotation=0,ha='right',va='center',fontsize=11)
        fig.suptitle(f'{task} / {arch} / block 2000\nCentered, unit-norm kernels; rows independently reordered by one-to-one matching. Shared colour scale ±{lim:.3f}.',fontsize=10)
        fig.subplots_adjust(left=.22,right=.99,top=.92,bottom=.02,wspace=.04,hspace=.2)
        fig.savefig(out/f'kernels_{task}_{arch}.png',dpi=150); plt.close(fig)
        fig,axes=plt.subplots(2,len(conditions),figsize=(17,7),sharex=True,sharey=True)
        for j,c in enumerate(conditions):
            for i,ep in enumerate([0,200]):
                w=examples[task,arch,c,ep];_,sim,_,cols=compare(w,templates,w)
                im=axes[i,j].imshow(sim,vmin=-1,vmax=1,cmap='RdBu_r')
                axes[i,j].scatter(cols,np.arange(16),s=9,c='black',marker='x')
                axes[i,j].set_title(f'{c}\nepoch {ep}',fontsize=9)
                axes[i,j].set_xlabel('Template index');axes[i,j].set_ylabel('Original channel index')
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label='Signed cosine; × = one-to-one assignment')
        fig.suptitle(f'{task} / {arch}: all 256 pairs, block 2000')
        fig.savefig(out/f'matrices_{task}_{arch}.png',dpi=150,bbox_inches='tight');plt.close(fig)


def plot_summary(df, conditions, out):
    settings=list(df[['task','architecture']].drop_duplicates().itertuples(index=False,name=None))
    fig,axes=plt.subplots(3,len(settings),figsize=(17,10),squeeze=False)
    for j,(task,arch) in enumerate(settings):
        subset=df[(df.task==task)&(df.architecture==arch)]
        for i,metric in enumerate(['nearest_cosine','assignment_cosine','matching_gap']):
            for c in conditions:
                g=subset[subset.condition==c].groupby('epoch')[metric].agg(['mean','std'])
                x=g.index.to_numpy();y=g['mean'].to_numpy();sd=g['std'].to_numpy()
                line,=axes[i,j].plot(x,y,label=c)
                axes[i,j].fill_between(x,y-sd,y+sd,alpha=.12,color=line.get_color())
            axes[i,j].set_ylabel(metric);axes[i,j].set_xlabel('Epoch')
            if i==0:axes[i,j].set_title(f'{task}\n{arch}')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=9)
    fig.suptitle('Kernel-to-template matching: all blocks; bands = sample SD, not confidence intervals')
    fig.tight_layout(rect=(0,.05,1,.95));fig.savefig(out/'matching_trajectories.png',dpi=160);plt.close(fig)
    final=df[df.epoch==200]
    fig,axes=plt.subplots(len(BEHAVIOR),len(settings),figsize=(17,15),squeeze=False)
    for j,(task,arch) in enumerate(settings):
        sub=final[(final.task==task)&(final.architecture==arch)]
        for i,target in enumerate(BEHAVIOR):
            for c in conditions:
                v=sub[sub.condition==c];axes[i,j].scatter(v.assignment_cosine,v[target],s=16,label=c,alpha=.8)
            axes[i,j].set_xlabel('One-to-one cosine');axes[i,j].set_ylabel(target)
            if i==0:axes[i,j].set_title(f'{task}\n{arch}')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=5,fontsize=9)
    fig.suptitle('Epoch 200: each dot is one block/model; colors are conditions. Descriptive, not causal associations.')
    fig.tight_layout(rect=(0,.035,1,.965));fig.savefig(out/'matching_vs_behavior.png',dpi=160);plt.close(fig)


def run(args):
    root=args.input.resolve();out=args.output.resolve()
    if out==root or root in out.parents:
        raise ValueError('Output must be outside the experimental input directory')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists(): (out/'COMPLETE.json').unlink()
    design=json.loads((root/'design.json').read_text())
    if digest(CORE)!=design['source_hashes']['core.py']:
        raise ValueError('Preserved core does not match source hash in design')
    if (design['epochs']!=200 or design['start_block']!=2000 or design['blocks']!=10
        or set(design['tasks'])!={'single_shape','two_concepts'}
        or set(design['architectures'])!={'TinyCNN','TwoLayerCNN'}
        or set(design['conditions'])!={'random','random_unitnorm','template_init','template_retention_1','template_release'}):
        raise ValueError('This protocol targets Stage B: 200 epochs, starting block 2000')
    spec=importlib.util.spec_from_file_location('preserved_release_core',CORE)
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    templates=core.bank()
    torch.set_num_threads(1)
    epochs=sorted(set([0,design['epochs']]+design['checkpoints']))
    rows=[];channels=[];matrices=[];examples={};provenance=[];errors=[]
    identity=list(itertools.product(design['tasks'],design['architectures'],range(2000,2000+design['blocks']),design['conditions']))
    for idx,(task,arch,block,condition) in enumerate(identity):
        folder=root/'runs'/task/arch/f'block{block:04d}'/condition
        initial=np.load(folder/'conv_initial.npy',allow_pickle=False)
        provenance.append(dict(path=str((folder/'config.json').relative_to(root)),sha256=digest(folder/'config.json')))
        config=json.loads((folder/'config.json').read_text())
        for key,value in dict(task=task,architecture=arch,block=block,condition=condition).items():
            if config[key]!=value:raise ValueError(f'Run identity mismatch: {folder}')
        for ep in epochs:
            cp=folder/f'epoch_{ep:04d}.pt'
            # Deliberately no weights_only=False fallback and no custom unpickling globals.
            saved=torch.load(cp,map_location='cpu',weights_only=True)
            if saved['epoch']!=ep:raise ValueError(f'Epoch mismatch: {cp}')
            w=saved['model']['conv.weight'].detach().cpu().numpy()
            if w.shape!=(16,1,9,9):raise ValueError(f'Unexpected first-layer shape: {w.shape}')
            if ep==0 and not np.array_equal(w,initial):raise ValueError('Initial weights differ')
            scores,sim,nearest,assignment=compare(w,templates,initial)
            row=dict(task=task,architecture=arch,block=block,condition=condition,epoch=ep,**scores)
            if ep:
                result_path=folder/'evaluations'/f'epoch_{ep:04d}'/'result.json'
                result=json.loads(result_path.read_text())
                for key in ['task','architecture','block','condition']:
                    if result[key]!=row[key]:raise ValueError(f'Metric identity mismatch: {result_path}')
                diff=abs(scores['nearest_cosine']-result['alignment']);errors.append(diff)
                if diff>1e-6:raise ValueError(f'Alignment mismatch {diff}: {cp}')
                row.update({k:result[k] for k in BEHAVIOR+['fidelity_selected','fidelity_random']})
                provenance.append(dict(path=str(result_path.relative_to(root)),sha256=digest(result_path)))
            matrices.append(sim)
            for ch in range(16):
                channels.append(dict(task=task,architecture=arch,block=block,condition=condition,epoch=ep,channel=ch,
                                     nearest_template=int(nearest[ch]),assigned_template=int(assignment[ch]),
                                     nearest_cosine=float(sim[ch,nearest[ch]]),assigned_cosine=float(sim[ch,assignment[ch]])))
            rows.append(row);provenance.append(dict(path=str(cp.relative_to(root)),sha256=digest(cp)))
            if block==2000 and ep in [0,200]:examples[task,arch,condition,ep]=w.copy()
        provenance.append(dict(path=str((folder/'conv_initial.npy').relative_to(root)),sha256=digest(folder/'conv_initial.npy')))
        if (idx+1)%10==0: print(f'Analyzed {idx+1}/{len(identity)} models',flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'per_checkpoint.csv',index=False)
    pd.DataFrame(channels).to_csv(out/'per_channel.csv',index=False)
    np.savez_compressed(out/'similarity_matrices.npz',cosine=np.stack(matrices),templates=templates,
                        row_index=np.arange(len(rows)))
    summary=df.groupby(['task','architecture','condition','epoch'])[METRICS].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(out/'summary.csv')
    correlations=[]
    for key,sub in df[df.epoch==200].groupby(['task','architecture','condition']):
        for metric in ['nearest_cosine','assignment_cosine','matching_gap']:
            for target in BEHAVIOR:
                valid=sub[[metric,target]].dropna()
                rho=float(spearmanr(valid[metric],valid[target]).statistic) if len(valid)>2 and valid.nunique().min()>1 else None
                correlations.append(dict(zip(['task','architecture','condition'],key),metric=metric,target=target,n=len(valid),spearman=rho))
    pd.DataFrame(correlations).to_csv(out/'within_condition_correlations.csv',index=False)
    plot_examples(examples,templates,design['conditions'],out);plot_summary(df,design['conditions'],out)
    final=df[df.epoch==200];text=['# Kernel similarity follow-up','',f'{len(identity)} models; {len(df)} checkpoints; all 256 kernel/template pairs at each checkpoint.',
        '',f'Maximum discrepancy from saved alignment: {max(errors):.3g}. No training or forward evaluations performed.',
        '', 'Exploratory morphology analysis. Similarity is not human interpretability. Correlations are within-condition descriptive summaries over ten blocks, not causal evidence.',
        '', '| Task / model | Condition | Nearest cosine | One-to-one cosine | Gap | Unique nearest templates |', '|---|---|---:|---:|---:|---:|']
    for key,g in final.groupby(['task','architecture','condition']):
        text.append('| '+' / '.join(key[:2])+' | '+key[2]+' | '+' | '.join(f'{g[m].mean():.4f} ± {g[m].std():.4f}' for m in ['nearest_cosine','assignment_cosine','matching_gap','nearest_template_count'])+' |')
    text+=['','![Matching trajectories](matching_trajectories.png)','','![Matching and behavior](matching_vs_behavior.png)',
           '', 'Gallery examples use block 2000 for every condition and both endpoints, with a shared centered/unit-norm colour scale. Raw norms are recorded separately. The redundant rank-10 bank limits diversity interpretations.',
           '', 'Matrices are indexed by per_checkpoint.csv row; per_channel.csv identifies both best and assigned template for each channel. Assignment maximizes total signed cosine; assignment_distance describes that same matching and is not a separately optimized Euclidean assignment.',
           '', 'The independent experimental submission audit remains pending.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    manifest=dict(design_sha256=digest(root/'design.json'),source_sha256=digest(Path(__file__)),core_sha256=digest(CORE),
                  plan_sha256=digest(Path(__file__).with_name('PLAN.md')),inputs=provenance,
                  versions=dict(torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__),
                  models=len(identity),checkpoints=len(df),max_alignment_error=max(errors),gallery_block=2000)
    (out/'COMPLETE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Completed. Report: {out / "REPORT.md"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=REPO/'results/retention_release_001')
    parser.add_argument('--output',type=Path,default=REPO/'analysis/kernel_similarity/results')
    run(parser.parse_args())
