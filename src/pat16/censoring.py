"""Conditional frame-selection experiments; never replace primary derivatives.

Randomization concerns frames within two fixed acquisitions, not participants.
The reported ranges are design-conditional resampling spreads, not confidence
intervals, edge-wise tests, or probabilities of a surgical effect.
"""
import json
from functools import lru_cache
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, ks_2samp, wasserstein_distance
from threadpoolctl import threadpool_limits
from . import ROOT, RESULTS
from .connectivity import clean_timeseries
from .nuisance import pca_signals, subspace_similarity
from .bold import sha

OUT = RESULTS/'censoring_investigation'
SESSIONS = ('preop', 'postop')
VERSION = 'censoring-matching-v1'
N = 68
CALIPER = .015
SEED = 3160
PROTOCOLS = {f'{pool}_{kind}': (pool, kind) for pool in ('primary', 'strict')
             for kind in ('duration', 'motion', 'motion_time')}
LABELS = {'duration':'Duration', 'motion':'Duration + FD',
          'motion_time':'Duration + FD + run half'}
STAGES = ['primary', 'restriction', 'coefficients', 'pca']


def random_pairs(fd_pre, fd_post, pool_pre, pool_post, rng, n=N,
                 caliper=CALIPER, time_halves=False):
    """Random-cost maximum-cardinality assignment, then n valid pairs.

    Pairing uses FD and optionally run half, never BOLD or coupling. This is
    not uniform sampling of either full pool: common-support frames are favored.
    """
    a, b = np.flatnonzero(pool_pre), np.flatnonzero(pool_post)
    allowed = abs(fd_pre[a, None]-fd_post[b][None, :]) <= caliper
    if time_halves:
        allowed &= (a[:, None]//90 == b[None, :]//90)
    cost = np.where(allowed, rng.uniform(size=allowed.shape), 10000.)
    i, j = linear_sum_assignment(cost)
    valid = allowed[i, j]
    i, j = i[valid], j[valid]
    if len(i) < n:
        raise RuntimeError(f'Only {len(i)} valid pairs, need {n}')
    chosen = rng.choice(len(i), n, replace=False)
    masks = [np.zeros_like(pool_pre, dtype=bool), np.zeros_like(pool_post, dtype=bool)]
    masks[0][a[i[chosen]]] = True
    masks[1][b[j[chosen]]] = True
    gaps = abs(fd_pre[a[i[chosen]]]-fd_post[b[j[chosen]]])
    return masks, float(gaps.max())


def frame_qc(keep, fd):
    idx = np.flatnonzero(keep)
    return {'frames':int(keep.sum()), 'retained_minutes':float(keep.sum()*2.4/60),
            'mean_retained_fd_mm':float(fd[keep].mean()),
            'median_retained_fd_mm':float(np.median(fd[keep])),
            'max_retained_fd_mm':float(fd[keep].max()),
            'span_minutes':float((idx[-1]-idx[0]+1)*2.4/60),
            'runs':int(1+np.sum(np.diff(idx)>1)),
            'adjacent_pairs':int(np.sum(np.diff(idx)==1)),
            **{f'third_{i+1}':int(np.sum((idx//60)==i)) for i in range(3)},
            'first_half':int(np.sum(idx<90))}


@lru_cache(maxsize=2)
def load_session(session):
    p = RESULTS/session
    tissue_path = OUT/session/'tissue_timeseries.npy'
    if not tissue_path.exists():
        raise RuntimeError('Run scripts/investigate_censoring.py to extract tissue cache')
    return dict(y=np.load(p/'roi_timeseries_raw.npy'), motion=np.loadtxt(p/'func_motion.tsv'),
                fd=np.load(p/'fd.npy'), sc=np.load(p/'sc_fraction.npy'),
                pcs=np.load(p/'primary_nuisance_pcs.npy'),
                residuals=np.load(p/'roi_timeseries_clean.npy'),
                primary=np.load(p/'fc_primary_keep.npy'), strict=np.load(p/'fc_fd03_keep.npy'),
                tissue=np.load(tissue_path))


def correlation(residuals, keep):
    fc=np.corrcoef(residuals[keep], rowvar=False)
    np.fill_diagonal(fc, 0.)
    return fc


def fit_selected(d, keep, mode='refit'):
    """Keep the original 180-frame temporal axis, including all censor gaps."""
    if mode == 'frozen':
        return correlation(d['residuals'], keep), None, None
    pcs = (d['pcs'] if mode=='coefficients' else
           pca_signals(d['tissue'].T, np.ones(d['tissue'].shape[1], bool), keep, True, n=5)[0])
    fc, residuals, _, qc, _ = clean_timeseries(d['y'], d['motion'], pcs, keep_override=keep)
    qc['fd_threshold_mm']=None
    qc['frame_mask_source']='explicit subset of eligible censor pool'
    return fc, qc, pcs


def coupling_measures(fc, sc, common, baseline_sc):
    upper=np.triu(np.ones(fc.shape, bool), 1)
    def corr(weights, support):
        k=upper & support & np.isfinite(fc)
        return float(spearmanr(np.log1p(weights[k]), fc[k]).statistic)
    return {'coupling':corr(sc, sc>0),
            'common_coupling':corr(sc, common),
            'fixed_pre_sc_coupling':corr(baseline_sc, common),
            'mean_fc':float(np.nanmean(fc[upper]))}


def prepare_cache():
    """Read existing masks and corrected BOLD without mutating spatial outputs."""
    OUT.mkdir(parents=True, exist_ok=True)
    lineage={}
    for session in SESSIONS:
        p=RESULTS/session; m=RESULTS/'nuisance_investigation'/session/'corrected'
        sources=[p/x for x in ['func_sdc.nii.gz','roi_timeseries_raw.npy','func_motion.tsv',
                              'fd.npy','sc_fraction.npy','primary_nuisance_pcs.npy',
                              'primary_pcs_fd03.npy','roi_timeseries_clean.npy',
                              'fc_primary_keep.npy','fc_fd03_keep.npy']]
        sources += [m/f'{t}_mask_p95.nii.gz' for t in ('wm','csf')]
        hashes={str(f.relative_to(ROOT)):sha(f) for f in sources}
        lineage.update(hashes)
        dest=OUT/session; dest.mkdir(exist_ok=True)
        meta=dest/'tissue_cache.json'
        if not meta.exists() or json.loads(meta.read_text()) != hashes:
            mask=(nib.load(sources[-2]).get_fdata()>.5)|(nib.load(sources[-1]).get_fdata()>.5)
            data=nib.load(p/'func_sdc.nii.gz').get_fdata(dtype=np.float32)
            np.save(dest/'tissue_timeseries.npy', data[mask].T)
            meta.write_text(json.dumps(hashes, indent=2))
        load_session.cache_clear()
    return lineage


def run(repeats=200):
    lineage=prepare_cache()
    code=[ROOT/'src/pat16'/n for n in ['censoring.py','connectivity.py','nuisance.py']]
    spec={'version':VERSION, 'seed':SEED, 'repeats':repeats, 'frames':N, 'fd_caliper_mm':CALIPER,
          'input_sha256':lineage, 'code_sha256':{p.name:sha(p) for p in code}}
    if (OUT/'design.json').exists() and json.loads((OUT/'design.json').read_text())==spec:
        print('Censoring experiment cache current', flush=True);return
    ds=[load_session(s) for s in SESSIONS]
    common=(ds[0]['sc']>0)&(ds[1]['sc']>0); pre_sc=ds[0]['sc']
    qc=[];decomp=[];subspaces=[]
    for session,d in zip(SESSIONS,ds):
        for pool in ('primary','strict'):
            qc.append({'session':session,'pool':pool,**frame_qc(d[pool],d['fd'])})
        for stage in STAGES:
            keep=d['primary'] if stage=='primary' else d['strict']
            if stage=='primary':fc=correlation(d['residuals'], keep);q=None;pcs=d['pcs']
            else:fc,q,pcs=fit_selected(d,keep,{'restriction':'frozen','coefficients':'coefficients','pca':'refit'}[stage])
            q=q or {'design_rank':39,'residual_df':int(d['primary'].sum()-39)}
            decomp.append({'session':session,'stage':stage,**coupling_measures(fc,d['sc'],common,pre_sc),
                           **frame_qc(keep,d['fd']),**q,
                           'df_meaning':'training residual df; subset effective df not estimated' if stage=='restriction' else 'selected fit algebraic residual df'})
            np.save(OUT/f'{session}_{stage}_fc.npy',fc)
        strict_pcs=np.load(RESULTS/session/'primary_pcs_fd03.npy')
        # Canonical correlations require centering on the comparison samples.
        k=d['strict']
        cs=subspace_similarity(d['pcs']-d['pcs'][k].mean(0),strict_pcs-strict_pcs[k].mean(0),k)
        subspaces.append({'session':session,'mean_squared_canonical_correlation':float(cs.mean()),
                          'squared_canonical_correlations':cs.tolist()})
    pd.DataFrame(qc).to_csv(OUT/'frame_qc.csv',index=False)
    pd.DataFrame(decomp).to_csv(OUT/'decomposition.csv',index=False)
    (OUT/'pc_subspaces.json').write_text(json.dumps(subspaces,indent=2))
    records=[];mask_archive={}
    # Each protocol has a deterministic independent random stream.
    for protocol_index,(protocol,(pool,kind)) in enumerate(PROTOCOLS.items()):
        rng=np.random.default_rng(SEED+protocol_index)
        masks=[]
        for rep in range(repeats):
            if kind=='duration':
                ks=[]
                for d in ds:
                    k=np.zeros(len(d['fd']),bool);k[rng.choice(np.flatnonzero(d[pool]),N,replace=False)]=True;ks.append(k)
                max_gap=np.nan
            else:
                ks,max_gap=random_pairs(ds[0]['fd'],ds[1]['fd'],ds[0][pool],ds[1][pool],rng,time_halves=kind=='motion_time')
            masks.append(ks)
            qa,qb=[frame_qc(k,d['fd']) for d,k in zip(ds,ks)]
            a,b=ds[0]['fd'][ks[0]],ds[1]['fd'][ks[1]]
            balance={'fd_mean_difference_mm':float(b.mean()-a.mean()),
                     'fd_abs_smd':float(abs(b.mean()-a.mean())/np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)),
                     'fd_ks':float(ks_2samp(a,b).statistic),
                     'fd_wasserstein_mm':float(wasserstein_distance(a,b)),
                     'maximum_pair_fd_gap_mm':max_gap,
                     'first_half_count_difference':qb['first_half']-qa['first_half']}
            for mode in ('frozen','refit'):
                for session,d,k,q in zip(SESSIONS,ds,ks,[qa,qb]):
                    fc,fitq,_=fit_selected(d,k,mode)
                    records.append({'protocol':protocol,'pool':pool,'matching':kind,'mode':mode,'replicate':rep,
                                    'session':session,**coupling_measures(fc,d['sc'],common,pre_sc),**q,**balance,
                                    'design_rank':np.nan if fitq is None else fitq['design_rank'],
                                    'residual_df':np.nan if fitq is None else fitq['residual_df']})
            if (rep+1)%50==0:print(f'{protocol}: {rep+1}/{repeats}',flush=True)
        mask_archive[protocol]=np.asarray(masks)
    np.savez_compressed(OUT/'matched_masks.npz',**mask_archive)
    r=pd.DataFrame(records);r.to_csv(OUT/'matched_replicates.csv',index=False)
    summary=[]
    for (protocol,mode),g in r.groupby(['protocol','mode'],sort=False):
        row={'protocol':protocol,'pool':g.pool.iloc[0],'matching':g.matching.iloc[0],'mode':mode,'repeats':repeats}
        for value in ['coupling','common_coupling','fixed_pre_sc_coupling','mean_fc']:
            wide=g.pivot(index='replicate',columns='session',values=value)
            delta=wide.postop-wide.preop
            row.update({f'{value}_delta_median':float(delta.median()),f'{value}_delta_q025':float(delta.quantile(.025)),
                        f'{value}_delta_q975':float(delta.quantile(.975)),f'{value}_positive_fraction':float((delta>0).mean()),
                        f'{value}_pre_median':float(wide.preop.median()),f'{value}_post_median':float(wide.postop.median())})
        for name in ['fd_mean_difference_mm','fd_abs_smd','fd_ks','fd_wasserstein_mm','maximum_pair_fd_gap_mm','first_half_count_difference']:
            vals=g[name].dropna()
            row[name+'_median']=float(vals.median()) if len(vals) else np.nan
            row[name+'_max_abs']=float(vals.abs().max()) if len(vals) else np.nan
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'matched_summary.csv',index=False)
    spec['repeats']=repeats
    (OUT/'design.json').write_text(json.dumps(spec,indent=2))
    print('Saved censoring experiments',flush=True)


def plot_results():
    import matplotlib.pyplot as plt
    dec=pd.read_csv(OUT/'decomposition.csv');r=pd.read_csv(OUT/'matched_replicates.csv')
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for session,color in zip(SESSIONS,['#245a81','#b35322']):
        d=dec[dec.session==session].set_index('stage').loc[STAGES]
        ax[0].plot(range(4),d.coupling,'o-',label=session,color=color)
    ax[0].set_xticks(range(4),['Primary','Strict frames\nfixed fit','+ coefficient\nrefit','+ PCA\nrefit'])
    ax[0].set_ylabel('SC–FC Spearman coupling');ax[0].legend();ax[0].set_title('A  |  Ordered diagnostic decomposition')
    w=dec.pivot(index='stage',columns='session',values='coupling').loc[STAGES]
    ax[1].bar(range(4),w.postop-w.preop,color=['#687e91','#98a9b7','#b68c66','#ba652f'])
    ax[1].axhline(0,color='black',lw=.7);ax[1].set_xticks(range(4),['Primary','Strict frames\nfixed fit','+ coefficient\nrefit','+ PCA\nrefit'])
    ax[1].set_ylabel('Post − pre coupling');ax[1].set_title('B  |  Direction depends on refitting stages')
    fig.tight_layout();fig.savefig(RESULTS/'figures/censoring_decomposition.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
    for ax,mode in zip(axes,['frozen','refit']):
        for i,(protocol,(pool,kind)) in enumerate(PROTOCOLS.items()):
            g=r[(r.protocol==protocol)&(r['mode']==mode)]
            w=g.pivot(index='replicate',columns='session',values='coupling');delta=w.postop-w.preop
            q=delta.quantile([.025,.5,.975]).values
            ax.plot(q[[0,2]],[i,i],lw=3,color='#245a81' if pool=='primary' else '#b35322')
            ax.plot(q[1],i,'o',color='black',ms=4)
        ax.axvline(0,color='black',lw=.8);ax.axvline(-.0207203,color='gray',ls='--',lw=.8)
        ax.set_yticks(range(6),[f'{p.title()}: {LABELS[k]}' for p,k in PROTOCOLS.values()])
        ax.set_xlabel('Post − pre SC–FC coupling');ax.set_title('Fixed primary fit' if mode=='frozen' else 'Refit PCA and regression per draw')
    axes[0].invert_yaxis();fig.suptitle('68 frames/session; dots = median, lines = central 95% resampling spread\nConditional on these runs and sampling designs; not confidence intervals',fontsize=11)
    fig.tight_layout();fig.savefig(RESULTS/'figures/censoring_matching.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,5.5),sharex='col')
    for row,session in enumerate(SESSIONS):
        d=load_session(session);t=np.arange(180)*2.4/60
        axs[row,0].plot(t,d['fd'],color='gray',lw=.8)
        for pool,color,h in [('primary','#245a81',.54),('strict','#b35322',.59)]:
            k=d[pool];axs[row,0].scatter(t[k],np.full(k.sum(),h),s=6,color=color,label=pool)
            axs[row,1].hist(d['fd'][k],bins=np.arange(0,.551,.025),histtype='step',density=True,color=color,label=pool)
        axs[row,0].axhline(.3,ls=':',color='#b35322');axs[row,0].axhline(.5,ls=':',color='#245a81')
        axs[row,0].set_ylabel(f'{session}\nFD (mm)');axs[row,0].set_ylim(0,.65);axs[row,1].set_ylabel('Density');axs[row,1].legend()
    axs[0,0].set_title('Frame selection (FD above 0.65 mm clipped)');axs[0,1].set_title('Motion among retained frames')
    axs[1,0].set_xlabel('Original run time (minutes)');axs[1,1].set_xlabel('Retained FD (mm)')
    fig.tight_layout();fig.savefig(RESULTS/'figures/censoring_frame_qc.png',dpi=180);plt.close(fig)


def show_draw(protocol='strict_motion', mode='refit', replicate=0):
    """Interactive notebook callback: recompute the selected saved draw."""
    import matplotlib.pyplot as plt
    ks=np.load(OUT/'matched_masks.npz')[protocol][replicate]
    fig,axs=plt.subplots(1,3,figsize=(12,3.3))
    with threadpool_limits(limits=1):
        for session,k,color in zip(SESSIONS,ks,['#245a81','#b35322']):
            d=load_session(session);fc,_,_=fit_selected(d,k,mode)
            axs[0].plot(np.arange(180)*.04,d['fd'],color=color,alpha=.25)
            axs[0].scatter(np.flatnonzero(k)*.04,d['fd'][k],color=color,s=10,label=session)
            axs[1].hist(d['fd'][k],bins=np.arange(.05,.55,.025),histtype='step',color=color,label=session)
            val=coupling_measures(fc,d['sc'],(d['sc']>0),d['sc'])['coupling']
            axs[2].scatter(session,val,color=color,s=55)
    axs[0].set(xlabel='Original time (min)',ylabel='FD (mm)',ylim=(0,.6));axs[0].legend()
    axs[1].set(xlabel='Selected FD (mm)',ylabel='Frames');axs[2].set_ylabel('SC–FC coupling')
    fig.suptitle(f'{protocol}, {mode}, saved draw {replicate}');fig.tight_layout()
    return fig
