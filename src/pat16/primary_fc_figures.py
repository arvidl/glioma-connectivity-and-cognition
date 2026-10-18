"""Figures showing observed points separately from conditional subset ranges."""
import numpy as np
import pandas as pd
from . import RESULTS
from .primary_fc import OUT,PRIMARY,FRACTIONS,MODES

LABELS={'half':'Half (73 frames)','two_thirds':'Two-thirds (97 frames)','four_fifths':'Four-fifths (116 frames)'}
COLORS={'frozen':'#4b8c89','coefficients':'#d59c3a','full_refit':'#425b9a'}

def intervals(ax,v,y,**kwargs):
    ax.errorbar(v['median'],y,xerr=np.vstack([v['median']-v.q025,v.q975-v['median']]),fmt='o',capsize=3,ms=4,**kwargs)

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=pd.read_csv(OUT/'summary.csv');network=pd.read_csv(OUT/'network_uncertainty.csv');sessions=pd.read_csv(OUT/'network_sessions.csv')
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for m,(mode,label) in enumerate(zip(MODES,['Frozen','Refit regression','Full refit'])):
        v=summary[(summary.protocol.str.endswith('block10'))&(summary.support=='fixed')&(summary['mode']==mode)].set_index('protocol').loc[[f+'_block10' for f in FRACTIONS]]
        intervals(axs[0],v,np.arange(3)+(m-1)*.18,label=label,color=COLORS[mode])
    axs[0].set_yticks(range(3),[LABELS[f] for f in FRACTIONS]);axs[0].set_title('Same duration in both sessions; block 10');axs[0].invert_yaxis()
    v=summary[(summary.protocol.str.startswith('two_thirds'))&(summary.support=='fixed')&(summary['mode']=='full_refit')]
    intervals(axs[1],v,np.arange(len(v)),color=COLORS['full_refit']);axs[1].set_yticks(range(len(v)),v.protocol.str.replace('two_thirds_block','Block '));axs[1].set_title('97 distinct frames; full nuisance refits');axs[1].invert_yaxis()
    for ax in axs:
        ax.axvline(0,color='gray',ls='--');ax.axvline(-.04172772719230558,color='black',ls=':',label='Observed full-run change');ax.set_xlabel('Post - pre SC-FC coupling change')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=8)
    fig.suptitle('Primary: 97 distinct frames, block 10, complete refits; conditional stability only')
    fig.savefig(RESULTS/'figures/primary_fc_joint.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4.7),layout='constrained')
    for ax,mod in zip(axs,['SC','FC']):
        v=network[(network.role=='primary')&(network.modality==mod)];y=np.arange(len(v))
        intervals(ax,v,y,color='#377d78' if mod=='SC' else '#425b9a',label='Subset median and range' if mod=='FC' else 'SC bootstrap median and range')
        ax.plot(v.observed_delta,y,'x',color='black',label='Observed full-run change');ax.set_yticks(y,v.network);ax.invert_yaxis();ax.axvline(0,color='gray',ls='--');ax.legend(fontsize=7)
        ax.set_title('SC: own-consensus support reselected' if mod=='SC' else 'FC: primary 97-frame full refits');ax.set_xlabel('Mean streamlines / 10,000 seeds' if mod=='SC' else 'Within-network mean Fisher z')
    fig.suptitle('Network changes: observed measurements and separate conditional perturbations')
    fig.savefig(RESULTS/'figures/primary_fc_network_changes.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(9,6),layout='constrained',sharex=True)
    for t,s in enumerate(['preop','postop']):
        sc=sessions[(sessions.modality=='SC')&(sessions.session==s)&(sessions.role=='primary')]
        fc=sessions[(sessions.modality=='FC')&(sessions.session==s)&(sessions.role=='primary')]
        x=np.arange(len(sc))+(t-.5)*.2;color=['#357da8','#d88b3c'][t]
        axs[0].plot(x,sc.observed,'o',color=color,label=s+' observed')
        axs[1].errorbar(x,fc['median'],yerr=np.vstack([fc['median']-fc.q025,fc.q975-fc['median']]),fmt='o',capsize=3,color=color,label=s+' subset median/range')
        axs[1].plot(x,fc.observed,'x',color=color,ms=7,label=s+' observed')
    axs[0].set_ylabel('Observed mean SC / 10,000 seeds');axs[1].set_ylabel('Mean FC Fisher z');axs[1].set_xticks(np.arange(len(fc)),fc.network)
    axs[0].legend(ncol=2,fontsize=8);axs[1].legend(ncol=2,fontsize=8);axs[1].axhline(0,color='gray',ls='--')
    fig.suptitle('Full-run network points; FC stability under refitted 97-frame block subsets')
    fig.savefig(RESULTS/'figures/primary_fc_network_sessions.png',dpi=180);plt.close(fig)
