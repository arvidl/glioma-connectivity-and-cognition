"""Publication figures from saved PC/sampling experiments; no recomputation."""
import numpy as np
import pandas as pd
from . import RESULTS
from .pc_sampling import OUT,DESIGNS,MECHANISMS

LABELS={'bootstrap':'Bootstrap (frequency weights)','unit_coverage':'Same coverage, unit weights','block_unique':'Nonoverlapping blocks, matched count','block80':'Nonoverlapping blocks, 80% retained'}
MLABELS={'coefficients':'Frozen PCs, refit regression','temporal_only':'Refit temporal projection only','preprocessing_only':'Refit preprocessing, frozen loadings','pca_only':'Frozen preprocessing, refit loadings','full_refit':'Refit preprocessing and loadings'}
COLORS=['#607b8d','#bb8b3e','#3d8f80','#77549b']


def intervals(ax,v,y,label=None,color=None):
    ax.errorbar(v['median'],y,xerr=np.vstack([v['median']-v.q025,v.q975-v['median']]),fmt='o',capsize=3,ms=4,label=label,color=color)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=pd.read_csv(OUT/'summary.csv');n=pd.read_csv(OUT/'network_ranges.csv');q=pd.read_csv(OUT/'fc_metrics.csv')
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for off,mode,color,label in [(-.2,'frozen','#4b8c89','Frozen'),(0,'coefficients','#d59c3a','Refit regression'),(.2,'full_refit','#425b9a','Refit PCs + regression')]:
        v=s[(s.support=='fixed')&(s['mode']==mode)].set_index('protocol').loc[[d+'_block10' for d in DESIGNS]]
        intervals(ax,v,np.arange(4)+off,label,color)
    ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(4),[LABELS[d] for d in DESIGNS]);ax.invert_yaxis()
    ax.set_xlabel('Post − pre SC–FC coupling Δ; conditional 2.5/97.5 percentiles')
    fig.legend(*ax.get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8)
    ax.set_title('Same SC bootstraps and fixed 823 edges; 10-frame sampling blocks')
    fig.savefig(RESULTS/'figures/pc_sampling_ranges.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,6.5),sharex=True,sharey=True,layout='constrained')
    for ax,design,color in zip(axs.ravel(),DESIGNS,COLORS):
        v=n[(n.protocol==design+'_block10')&(n['mode']=='full_refit')]
        y=np.arange(len(v));intervals(ax,v,y,color=color);ax.plot(v.observed_delta,y,'x',color='black',label='Observed Δ')
        ax.set_yticks(y,v.network);ax.axvline(0,color='gray',ls='--');ax.set_title(LABELS[design],fontsize=10)
        ax.set_xlabel('Post − pre within-network mean Fisher z')
    axs[0,0].invert_yaxis()
    fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside lower center',fontsize=8)
    fig.suptitle('Full nuisance refits: empirical centers and ranges depend on sampling design')
    fig.savefig(RESULTS/'figures/pc_sampling_networks.png',dpi=180);plt.close(fig)
    m=pd.read_csv(OUT/'mechanism_summary.csv');metrics=pd.read_csv(OUT/'mechanism_metrics.csv')
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    v=m.set_index('mode').loc[list(MECHANISMS)];y=np.arange(5)
    intervals(axs[0],v,y,color='#425b9a');axs[0].set_yticks(y,[MLABELS[k] for k in MECHANISMS]);axs[0].axvline(0,color='gray',ls='--');axs[0].invert_yaxis()
    axs[0].set(xlabel='Post − pre coupling Δ',title='Same frequency-weighted block-10 draws')
    for offset,sess,color in [(-.12,'preop','#397e89'),(.12,'postop','#bd8841')]:
        rows=[]
        for mode in MECHANISMS:
            a=metrics[(metrics['mode']==mode)&(metrics.session==sess)].cortical_mean_PC_removed
            lo,med,hi=np.quantile(a,[.025,.5,.975]);rows.append(dict(median=med,q025=lo,q975=hi))
        intervals(axs[1],pd.DataFrame(rows),y+offset,sess,color)
    axs[1].set_yticks(y,[]);axs[1].invert_yaxis();axs[1].set(xlabel='Fraction of cortical-mean variance removed by PCs',title='Relative to the same draw’s background GLM')
    axs[1].legend(fontsize=8);fig.suptitle('Training controls separate temporal projection, voxel scaling and PCA loadings')
    fig.savefig(RESULTS/'figures/pc_sampling_mechanism.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,column,title in zip(axs,['adjusted_PC_cos2_min','gap5_relative'],['Minimum PC-span cosine² after background projection','Relative eigenvalue gap at PC 5/6']):
        for offset,sess,color in [(-.12,'preop','#397e89'),(.12,'postop','#bd8841')]:
            rows=[]
            for design in DESIGNS:
                a=q[(q.protocol==design+'_block10')&(q['mode']=='full_refit')&(q.session==sess)][column]
                lo,med,hi=np.quantile(a,[.025,.5,.975]);rows.append(dict(median=med,q025=lo,q975=hi))
            intervals(ax,pd.DataFrame(rows),np.arange(4)+offset,sess,color)
        ax.set_yticks(range(4),[LABELS[d] for d in DESIGNS] if ax is axs[0] else []);ax.invert_yaxis();ax.set_title(title,fontsize=9);ax.set_xlabel(column)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=8)
    fig.savefig(RESULTS/'figures/pc_sampling_stability.png',dpi=180);plt.close(fig)
