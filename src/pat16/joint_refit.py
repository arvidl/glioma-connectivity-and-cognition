"""Paired frozen / regression-only / complete nuisance refits of saved blocks.

Multiplicities weight original-time rows. They never create new independent
observations, temporal derivatives or DCT positions. Sparse fits fail openly.
"""
import json,time
import numpy as np
import pandas as pd
from scipy.linalg import eigh,lstsq,subspace_angles,orth
from threadpoolctl import threadpool_limits
from . import ROOT,RESULTS,SESSIONS
from .bold import sha
from .censoring import load_session
from .nuisance import temporal_basis,pca_signals
from .joint_sampling import OUT as PREVIOUS,I,J,B,F,evaluate,spread,variance_components

OUT=RESULTS/'joint_refit'
VERSION='full-run-frequency-weighted-refit-v1'
MODES=('frozen','coefficients','full_refit')
PROTOCOLS=tuple(f'{pool}_block{b}' for pool in ('primary','strict') for b in (5,10,20))
MIN_UNIQUE,MIN_RESIDUAL,MIN_PAIRED=60,15,100


def multiplicities(idx,nt):
    idx=np.asarray(idx)
    if idx.ndim!=1 or idx.dtype.kind not in 'iu' or (idx<0).any() or (idx>=nt).any():
        raise ValueError('Original-time integer indices required')
    return np.bincount(idx,minlength=nt).astype(float)


def weighted_pcs(tissue,idx,n=5):
    """Exactly the duplicated-row PCA estimator, via weighted unique rows.

    Refit temporal projection, voxel mean/SD and PCA. Return all original-time
    scores so the regression design retains its original temporal coordinates.
    """
    w=multiplicities(idx,len(tissue));use=w>0;root=np.sqrt(w[use])[:,None]
    t=temporal_basis(len(tissue))
    beta=lstsq(t[use]*root,np.asarray(tissue[use],float)*root,cond=1e-8)[0]
    z=np.asarray(tissue,float)-t@beta
    z-=np.average(z,axis=0,weights=w)
    sd=np.sqrt(np.average(z*z,axis=0,weights=w));valid=sd>1e-5
    if valid.sum()<n:raise RuntimeError('Insufficient varying tissue voxels')
    z=z[:,valid]/sd[valid];zw=z[use]*root
    eigen,u=eigh(zw@zw.T);order=np.argsort(eigen)[::-1]
    eigen=np.maximum(eigen[order],0);u=u[:,order]
    if (eigen>eigen[0]*1e-12).sum()<n:raise RuntimeError('Insufficient tissue PCA rank')
    scores=z@(zw.T@(u[:,:n]/np.sqrt(eigen[:n])))
    return scores,eigen[:n]/eigen.sum(),int(valid.sum())


def regression_design(motion,pcs,w):
    nt=len(motion);t=np.arange(nt)
    dm=np.diff(motion,axis=0,prepend=motion[:1])
    k=np.arange(1,nt);k=k[k/(2*nt*2.4)<.01]
    x=np.column_stack([t,motion,dm,motion**2,dm**2,pcs,
        np.cos(np.pi*(t[:,None]+.5)*k/nt)])
    x-=np.average(x,axis=0,weights=w)
    sd=np.sqrt(np.average(x*x,axis=0,weights=w));good=sd>1e-8
    return np.column_stack([np.ones(nt),x[:,good]/sd[good]])


def weighted_correlation(y,w):
    z=y-np.average(y,axis=0,weights=w)
    c=(z*w[:,None]).T@z;sd=np.sqrt(np.diag(c))
    if not np.isfinite(c).all() or (sd<=1e-8).any():raise RuntimeError('Degenerate ROI residuals')
    c=c/sd[:,None]/sd[None,:];np.fill_diagonal(c,0.)
    return c


def refit(d,idx,pcs=None):
    """Refit on frequency-weighted original-time rows; diagnose unique support.

    Apply conservative eligibility to DISTINCT rows: >=60 and >=15 residual
    directions after design rank. These are feasibility guards, not effective
    sample size estimates for autocorrelated BOLD. Never rescue a failed draw
    by counting duplicates as new information or redrawing more favorable ones.
    """
    w=multiplicities(idx,len(d['y']));use=w>0
    if use[:4].any():raise ValueError('Initial censored frames were resampled')
    if pcs is None:
        pcs,variance,nvox=weighted_pcs(d['tissue'],idx)
    else:variance=np.full(5,np.nan);nvox=np.nan
    x=regression_design(d['motion'],pcs,w);root=np.sqrt(w[use])[:,None]
    beta,_,solver_rank,singular=lstsq(x[use]*root,d['y'][use]*root,cond=1e-8)
    residuals=d['y']-x@beta
    rank=int(np.linalg.matrix_rank(x[use]));unique=int(use.sum())
    reasons=[]
    if unique<MIN_UNIQUE:reasons.append('fewer_than_60_distinct_rows')
    if unique-rank<MIN_RESIDUAL:reasons.append('fewer_than_15_distinct_residual_directions')
    if int(solver_rank)!=rank:reasons.append('solver_rank_disagrees')
    qc=dict(sample_rows=len(idx),unique_frames=unique,design_rank=rank,solver_rank=int(solver_rank),
        rows_minus_rank=len(idx)-rank,distinct_rows_minus_rank=unique-rank,
        eligible=not reasons,reason=';'.join(reasons) if reasons else 'eligible',
        design_condition=float(singular[0]/singular[-1]) if singular[-1]>0 else np.inf,
        tissue_voxels=nvox,PC_variance_fraction_sum=float(np.sum(variance)),
        minimum_ROI_residual_SD=float(np.min(np.sqrt(np.average(residuals**2,axis=0,weights=w)))))
    fc=np.full((100,100),np.nan)
    if qc['eligible']:
        try:fc=weighted_correlation(residuals,w)
        except RuntimeError as e:qc.update(eligible=False,reason=str(e))
    return fc,qc,pcs,residuals


def inputs():
    paths=[ROOT/'src/pat16/joint_refit.py',ROOT/'config/JOINT_REFIT.md',
        PREVIOUS/'design.json',PREVIOUS/'block_indices.npz',PREVIOUS/'fc_draws.npy',
        PREVIOUS/'sc_weights.npy',PREVIOUS/'sc_selected_support.npy',PREVIOUS/'sc_run_indices.npy',
        RESULTS/'primary_sc/policy.json']
    for s in SESSIONS:
        paths += [RESULTS/s/name for name in ['roi_timeseries_raw.npy','func_motion.tsv','fd.npy','primary_nuisance_pcs.npy','fc_primary.npy','fc_fd03.npy','fc_primary_keep.npy','fc_fd03_keep.npy']]
        paths += [RESULTS/'censoring_investigation'/s/'tissue_timeseries.npy']
        paths += [RESULTS/'primary_sc'/s/name for name in ['common_support.npy','common_per_10000_seeds.npy']]
    paths += [ROOT/'src/pat16'/f'{name}.py' for name in ['joint_sampling','censoring','nuisance','connectivity']]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def analyze(force=False):
    OUT.mkdir(exist_ok=True);lineage=inputs();dp=OUT/'design.json'
    if dp.exists() and not force:
        old=json.loads(dp.read_text())
        if old['inputs']==lineage and all(sha(OUT/p)==h for p,h in old['outputs'].items()):
            print('Full-run refit checkpoint verified',flush=True);return
    started=time.time();source=json.loads((PREVIOUS/'design.json').read_text())
    oldfc=np.load(PREVIOUS/'fc_draws.npy',mmap_mode='r');indices=np.load(PREVIOUS/'block_indices.npz')
    fc=np.lib.format.open_memmap(OUT/'fc_draws.npy',mode='w+',dtype='float64',shape=(6,3,2,F,len(I)))
    fc[:]=np.nan;records=[]
    with threadpool_limits(limits=1):
        for p,protocol in enumerate(PROTOCOLS):
            print('Frequency-weighted refits:',protocol,flush=True)
            op=source['protocols'].index(protocol);pool=protocol.split('_')[0]
            for t,s in enumerate(SESSIONS):
                d=load_session(s);basepcs=d['pcs']
                if pool=='strict':
                    basepcs=pca_signals(d['tissue'].T,np.ones(d['tissue'].shape[1],bool),d['strict'],True,n=5)[0]
                fc[p,0,t]=oldfc[op,t]
                for f,idx in enumerate(indices[f'{protocol}_{s}']):
                    assert d[pool][idx].all()
                    for m,pcs in [(1,basepcs),(2,None)]:
                        fit,q,scores,_=refit(d,idx,pcs)
                        fc[p,m,t,f]=fit[I,J]
                        w=multiplicities(idx,180);use=w>0;root=np.sqrt(w[use])[:,None]
                        similarity=np.cos(subspace_angles(orth(basepcs[use]*root),orth(scores[use]*root)))**2
                        records.append(dict(protocol=protocol,mode=MODES[m],session=s,draw=f,
                            mean_fd=float(np.average(d['fd'],weights=w)),
                            minimum_PC_subspace_cos2=float(similarity.min()),**q))
            fc.flush()
    q=pd.DataFrame(records);q.to_csv(OUT/'fit_qc.csv',index=False)
    weights=np.load(PREVIOUS/'sc_weights.npy');selected=np.load(PREVIOUS/'sc_selected_support.npy')
    fixed=np.load(RESULTS/'primary_sc/preop/common_support.npy')[I,J]
    point=np.asarray([np.load(RESULTS/'primary_sc'/s/'common_per_10000_seeds.npy')[I,J] for s in SESSIONS])
    summary,contrasts,feasible,halves,session_rows=[],[],[],[],[]
    with threadpool_limits(limits=1):
        for p,protocol in enumerate(PROTOCOLS):
            valid=np.isfinite(fc[p,1:]).all(axis=(0,1,3))
            draws=np.flatnonzero(valid)
            row=dict(protocol=protocol,requested=F,paired_eligible=len(draws),minimum_for_summary=MIN_PAIRED,
                summary_available=len(draws)>=MIN_PAIRED)
            for s in SESSIONS:
                v=q[(q.protocol==protocol)&(q.session==s)&(q['mode']=='full_refit')]
                row[s+'_eligible']=int(v.eligible.sum());row[s+'_unique_min']=int(v.unique_frames.min())
                row[s+'_unique_median']=float(v.unique_frames.median());row[s+'_unique_max']=int(v.unique_frames.max())
            feasible.append(row)
            np.save(OUT/f'{protocol}_paired_draws.npy',draws)
            if len(draws)<MIN_PAIRED:continue
            for support,mask in [('fixed',fixed),('reselected',selected)]:
                print('Crossing paired nuisance models:',protocol,support,flush=True)
                delta,rhos=[],[]
                for m,mode in enumerate(MODES):
                    fcv=fc[p,m][:,draws,:]
                    rho=evaluate(weights,mask,fcv);d=rho[1]-rho[0]
                    point_rho=evaluate(point[:,None,:],fixed,fcv)[:,0,:]
                    delta.append(d);rhos.append(rho)
                    base=dict(protocol=protocol,support=support,mode=mode,FC_draws=len(draws),SC_draws=B)
                    summary.append(base|spread(d)|variance_components(d))
                    for t,s in enumerate(SESSIONS):session_rows.append(base|{'session':s}|spread(rho[t]))
                    for label,a in [('SC_first',d[:B//2]),('SC_second',d[B//2:]),('FC_first',d[:,:len(draws)//2]),('FC_second',d[:,len(draws)//2:])]:
                        halves.append(base|{'half':label}|spread(a))
                    np.savez_compressed(OUT/f'{protocol}_{support}_{mode}.npz',rho=rho,delta=d,fc_only_rho=point_rho,FC_draw_ids=draws)
                for tag,a in [('coefficients_minus_frozen',delta[1]-delta[0]),('full_minus_coefficients',delta[2]-delta[1]),('full_minus_frozen',delta[2]-delta[0])]:
                    contrasts.append(dict(protocol=protocol,support=support,contrast=tag)|spread(a))
                np.savez_compressed(OUT/f'{protocol}_{support}_paired_shifts.npz',
                    coefficients_minus_frozen=delta[1]-delta[0],full_minus_coefficients=delta[2]-delta[1],full_minus_frozen=delta[2]-delta[0])
    pd.DataFrame(feasible).to_csv(OUT/'feasibility.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'paired_shifts.csv',index=False)
    pd.DataFrame(session_rows).to_csv(OUT/'session_coupling.csv',index=False)
    pd.DataFrame(halves).to_csv(OUT/'monte_carlo_halves.csv',index=False)
    figures()
    output={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='design.json'}
    dp.write_text(json.dumps(dict(version=VERSION,inputs=lineage,outputs=output,
        protocols=PROTOCOLS,modes=MODES,SC_draws=B,FC_draws=F,MIN_UNIQUE=MIN_UNIQUE,
        MIN_DISTINCT_RESIDUAL_DIRECTIONS=MIN_RESIDUAL,MIN_PAIRED=MIN_PAIRED,
        frozen_and_refitted_draws_are_paired=True,seconds=time.time()-started),indent=2))
    print('Full-run refit analysis completed in',round(time.time()-started,1),'seconds',flush=True)


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=pd.read_csv(OUT/'summary.csv');q=pd.read_csv(OUT/'feasibility.csv')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),sharex=True,layout='constrained')
    for ax,support in zip(axs,('fixed','reselected')):
        for offset,mode,color in [(-.2,'frozen','#4b8c89'),(0,'coefficients','#d59c3a'),(.2,'full_refit','#425b9a')]:
            v=d[(d.support==support)&(d['mode']==mode)].set_index('protocol').reindex(PROTOCOLS[:3]).reset_index()
            y=np.arange(len(v))+offset
            ax.errorbar(v['median'],y,xerr=np.array([v['median']-v.q025,v.q975-v['median']]),fmt='o',capsize=3,label={'frozen':'Frozen','coefficients':'Refit coefficients','full_refit':'Refit PCs + coefficients'}[mode],color=color,ms=4)
        ax.set_yticks(range(len(v)),['Block 5 (12 s)','Block 10 (24 s)','Block 20 (48 s)']);ax.invert_yaxis();ax.axvline(0,color='gray',ls='--')
        ax.set_xlabel('Post − pre coupling Δ; conditional range');ax.set_title(support+' consensus support')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8);fig.suptitle('Paired full-run primary sampling: nuisance refitting and SC support')
    fig.savefig(RESULTS/'figures/joint_refit_ranges.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained')
    y=np.arange(len(q));ax.barh(y,q.paired_eligible,color='#425b9a')
    ax.axvline(MIN_PAIRED,color='#d59c3a',ls='--',label='Minimum for a range summary')
    ax.set_yticks(y,q.protocol);ax.invert_yaxis();ax.set_xlim(0,210)
    ax.set_xlabel('Eligible paired FC draws / 200');ax.legend(fontsize=8)
    fig.savefig(RESULTS/'figures/joint_refit_feasibility.png',dpi=180);plt.close(fig)

