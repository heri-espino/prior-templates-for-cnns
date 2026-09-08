"""Summarize completed and partial runs without choosing checkpoints from test results."""
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def summarize(root):
    root=Path(root);dest=root/'analysis';dest.mkdir(exist_ok=True)
    rows=[];evaluations=[];endpoints=[]
    design=json.loads((root/'design.json').read_text())
    for f in sorted((root/'runs').glob('*/*/block*/*/history.json')):
        task,arch,block,condition=f.relative_to(root/'runs').parts[:4]
        identity=dict(task=task,architecture=arch,block=block,condition=condition)
        h=json.loads(f.read_text())
        if not h:continue
        rows.extend(dict(**identity,**r) for r in h)
        reached=[r['epoch'] for r in h if r['val_acc']>=.95]
        endpoints.append(dict(**identity,complete=(f.parent/'complete.json').exists(),epochs_observed=len(h),early_mean_val_acc=np.mean([r['val_acc'] for r in h if r['epoch']<=10]) if len(h)>=10 else None,attained_95=bool(reached),first_epoch_95=min(reached) if reached else None,last_val_acc=h[-1]['val_acc'],last_val_ce=h[-1]['val_ce'],late_val_ce_slope=float(np.polyfit([r['epoch'] for r in h[-20:]],[r['val_ce'] for r in h[-20:]],1)[0]) if len(h)>=20 else None))
        for ev in sorted((f.parent/'evaluations').glob('epoch_*/result.json')):
            evaluations.append(dict(**json.loads(ev.read_text()),epoch=int(ev.parent.name.split('_')[1])))
    if not rows:return
    df=pd.DataFrame(rows);df.to_csv(dest/'learning_curves.csv',index=False)
    pd.DataFrame(evaluations).to_csv(dest/'checkpoint_metrics.csv',index=False)
    ed=pd.DataFrame(endpoints);ed.to_csv(dest/'per_run.csv',index=False)
    settings=list(df[['task','architecture']].drop_duplicates().itertuples(index=False,name=None))
    fig,axes=plt.subplots(3,len(settings),figsize=(5*len(settings),10),squeeze=False)
    for j,(task,arch) in enumerate(settings):
        subset=df[(df.task==task)&(df.architecture==arch)]
        for condition,s in subset.groupby('condition',sort=False):
            for i,key in enumerate(['val_acc','val_ce','alignment']):
                g=s.groupby('epoch')[key];mean=g.mean();sd=g.std().fillna(0)
                axes[i,j].plot(mean.index,mean,label=condition)
                axes[i,j].fill_between(mean.index,(mean-sd).to_numpy(),(mean+sd).to_numpy(),alpha=.12)
                axes[i,j].set_ylabel(key);axes[i,j].set_xlabel('Epoch')
        axes[0,j].set_title(task+' / '+arch);axes[0,j].set_ylim(0,1.02)
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3)
    fig.suptitle('Mean ± SD across available blocks; partial runs may have unequal counts')
    fig.tight_layout(rect=(0,.07,1,.96));fig.savefig(dest/'learning_curves.png',dpi=160);plt.close(fig)
    checkpoint_plot=False
    evdf=pd.DataFrame(evaluations)
    if len(evdf) and all(k in evdf for k in ['semantic_auc','localization_iou','causal_usefulness']):
        fig,axes=plt.subplots(3,len(settings),figsize=(5*len(settings),10),squeeze=False)
        for j,(task,arch) in enumerate(settings):
            subset=evdf[(evdf.task==task)&(evdf.architecture==arch)]
            for condition,s in subset.groupby('condition',sort=False):
                for i,key in enumerate(['semantic_auc','localization_iou','causal_usefulness']):
                    g=s.groupby('epoch')[key];mean=g.mean();sd=g.std().fillna(0)
                    axes[i,j].plot(mean.index,mean,label=condition)
                    axes[i,j].fill_between(mean.index,(mean-sd).to_numpy(),(mean+sd).to_numpy(),alpha=.12)
                    axes[i,j].set_ylabel(key);axes[i,j].set_xlabel('Checkpoint epoch')
            axes[0,j].set_title(task+' / '+arch)
        h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3)
        fig.suptitle('Independent concept and causal diagnostics: mean ± SD; descriptive')
        fig.tight_layout(rect=(0,.07,1,.96));fig.savefig(dest/'checkpoint_metrics.png',dpi=160);plt.close(fig)
        checkpoint_plot=True
    lines=['# Retention-release experiment results','',f"Completed {int(ed.complete.sum())} of {design['blocks']*len(design['tasks'])*len(design['architectures'])*len(design['conditions'])} planned runs. Observed {len(df)} epoch records.",'','![Learning curves](learning_curves.png)','','These are descriptive results. Compare paired blocks; do not treat epochs or checkpoints as independent replicates. Non-attainment of 95% is recorded explicitly, not replaced with the final epoch. Test outcomes must not select a preferred release schedule or checkpoint.','', '| Task / model | Condition | Completed | Mean final validation accuracy | Reached 95% |','|---|---|---:|---:|---:|']
    for (t,a,c),s in ed.groupby(['task','architecture','condition']):
        complete=s[s.complete]
        final=f'{100*complete.last_val_acc.mean():.2f}%' if len(complete) else 'pending'
        lines.append(f'| {t} / {a} | {c} | {len(complete)} | {final} | {int(s.attained_95.sum())}/{len(s)} observed |')
    lines+=['','Raw tables: `learning_curves.csv`, `per_run.csv`, `checkpoint_metrics.csv`. Selected-channel causal usefulness, concept AUROC, localization, and alternative classifier diagnostics are measured at scheduled checkpoints. Inspect those alongside accuracy and alignment; kernel resemblance alone is not semantic interpretability.']
    if checkpoint_plot: lines+=['','![Checkpoint diagnostics](checkpoint_metrics.png)']
    (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Report saved:',dest/'REPORT.md',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',nargs='?',default='outputs/main');summarize(p.parse_args().output)
