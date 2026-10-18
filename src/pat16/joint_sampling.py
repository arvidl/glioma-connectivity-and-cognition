"""Cross whole-run SC bootstraps with temporally coherent / matched FC draws.

Conditional computational uncertainty only. No edgewise resampling, arbitrary
pre/post tractogram pairing, independent-cell standard errors or clinical CIs.
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from . import ROOT, RESULTS, SESSIONS
from .bold import sha
from .censoring import OUT as CENSOR, PROTOCOLS, load_session, fit_selected
from .primary_sc import OUT as PRIMARY, MIN_HITS, SEEDS

OUT = RESULTS/'joint_sampling'
VERSION = 'joint-run-bootstrap-matched-fc-v1'
B, F, RNG_SEED = 500, 200, 4160
BLOCKS = (5, 10, 20)
SUPPORTS = ('fixed', 'reselected')
I, J = np.triu_indices(100, 1)


def standardized_ranks(x):
    """Average ties, each row centered and unit Euclidean norm."""
    r = rankdata(np.asarray(x), axis=-1, method='average')
    r -= r.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(r, axis=-1, keepdims=True)
    if not np.isfinite(r).all() or (norm == 0).any():
        raise ValueError('Nonfinite or constant rank vector')
    return r/norm


def sc_bootstrap(counts, indices, min_hits=MIN_HITS):
    """Each independent session draw resamples 20 entire seed runs, with zeros.

    counts: (2,20,edges); indices: (2,B,20). Whole-run clustering preserves
    dependence among all edges within a tractogram. No individual edge draws.
    """
    counts, indices = np.asarray(counts), np.asarray(indices)
    if counts.ndim != 3 or counts.shape[:2] != (2, 20):
        raise ValueError('Need two sessions with twenty runs each')
    if indices.ndim != 3 or indices.shape[0] != 2 or indices.shape[2] != 20:
        raise ValueError('Need twenty sampled run indices per bootstrap/session')
    if (indices < 0).any() or (indices >= 20).any():
        raise ValueError('Run index outside 0..19')
    weights, hits = [], []
    for t in range(2):
        multiplicities = np.eye(20, dtype=np.int16)[indices[t]].sum(axis=1)
        weights.append((multiplicities@counts[t])/20/SEEDS*10000)
        hits.append(multiplicities@(counts[t] >= 1).astype(np.int16))
    weights, hits = np.asarray(weights), np.asarray(hits)
    return weights, (hits >= min_hits).all(axis=0), hits


def block_indices(keep, block, rng):
    """Circular blocks on the ORIGINAL axis; omit censored members inside blocks.

    Continue drawing blocks until the original retained count is collected,
    then truncate. This fixes retained exposure within each session, while
    allowing repeated frames. Never concatenate retained frames before blocking.
    """
    keep = np.asarray(keep, bool)
    n, target = len(keep), int(keep.sum())
    if target < 3 or not 1 <= block <= n:
        raise ValueError('Invalid block length / censor set')
    parts, total = [], 0
    while total < target:
        idx = (rng.integers(n)+np.arange(block)) % n
        idx = idx[keep[idx]]
        parts.append(idx); total += len(idx)
    return np.concatenate(parts)[:target]


def variance_components(d):
    """Exact population-variance decomposition of this finite crossed grid.

    The two axes have independent product resampling weights. Shares quantify
    this computational experiment, not independent biological sources of error.
    """
    d = np.asarray(d)
    grand = d.mean()
    sc = d.mean(axis=1, keepdims=True)-grand
    fc = d.mean(axis=0, keepdims=True)-grand
    interaction = d-grand-sc-fc
    components = np.array([np.mean(sc**2), np.mean(fc**2), np.mean(interaction**2)])
    total = float(np.var(d))
    if not np.isclose(components.sum(), total, rtol=1e-10, atol=1e-15):
        raise AssertionError('Crossed variance identity failed')
    return dict(zip(('SC_variance', 'FC_variance', 'interaction_variance'), components)) | {
        'total_variance': total, **dict(zip(('SC_share', 'FC_share', 'interaction_share'), components/total))}


def evaluate(weights, support, fc):
    """rho: (2,B,F), each FC/SC reranked on its actual common support."""
    if not np.isfinite(fc).all():
        raise ValueError('This delivered design requires all 100 FC nodes finite')
    b = weights.shape[1]
    rho = np.empty((2, b, fc.shape[1]))
    if support.ndim == 1:
        for t in range(2):
            sr = standardized_ranks(weights[t][:, support])
            fr = standardized_ranks(fc[t][:, support])
            rho[t] = sr@fr.T
    else:
        for k in range(b):
            mask = support[k]
            if mask.sum() < 3 or (weights[:, k, mask] <= 0).any():
                raise ValueError('Invalid selected SC support')
            for t in range(2):
                sr = standardized_ranks(weights[t, k, mask])
                fr = standardized_ranks(fc[t][:, mask])
                rho[t, k] = fr@sr
    return rho


def lineage():
    paths = [ROOT/'src/pat16'/x for x in ['joint_sampling.py','censoring.py','nuisance.py','connectivity.py','primary_sc.py']]
    paths += [ROOT/'config/JOINT_SAMPLING.md', CENSOR/'matched_masks.npz', PRIMARY/'policy.json']
    for s in SESSIONS:
        paths += [RESULTS/'sc_reliability'/s/'counts.npy', PRIMARY/s/'common_per_10000_seeds.npy']
        paths += [RESULTS/s/x for x in ['roi_timeseries_raw.npy','roi_timeseries_clean.npy','func_motion.tsv','fd.npy','fc_primary_keep.npy','fc_fd03_keep.npy','primary_nuisance_pcs.npy','fc_primary.npy','fc_fd03.npy']]
        paths += [CENSOR/s/'tissue_timeseries.npy']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def fc_draws():
    """Archive every FC matrix, matched mask and block index; use separate RNGs."""
    from numpy.lib.format import open_memmap
    protocols = list(PROTOCOLS)+[f'{pool}_block{block}' for pool in ('primary','strict') for block in BLOCKS]
    path = OUT/'fc_draws.npy'
    fc = open_memmap(path, mode='w+', dtype='float64', shape=(len(protocols), 2, F, len(I)))
    masks = np.load(CENSOR/'matched_masks.npz')
    records, indices = [], {}
    for p, protocol in enumerate(protocols):
        print('FC draws:', protocol, flush=True)
        for t, s in enumerate(SESSIONS):
            d = load_session(s)
            if protocol in PROTOCOLS:
                for f, pair in enumerate(masks[protocol]):
                    keep = pair[t]
                    # Strict pre-op has exactly 68 eligible frames: cache identical fits.
                    cache_key = keep.tobytes()
                    if f == 0: fits = {}
                    if cache_key not in fits:
                        fit, qc, _ = fit_selected(d, keep)
                        fits[cache_key] = fit[I, J], qc
                    vec, qc = fits[cache_key]
                    fc[p, t, f] = vec
                    records.append(dict(protocol=protocol, session=s, draw=f,
                        retained=len(np.flatnonzero(keep)), unique_frames=int(keep.sum()),
                        design_rank=qc['design_rank'], residual_df=qc['residual_df'],
                        mean_fd=float(d['fd'][keep].mean()), nuisance='fully_refit'))
            else:
                pool, block = protocol.split('_block'); block = int(block)
                keep = d[pool]
                if pool == 'primary': residuals = d['residuals']
                else:
                    fit, residuals = strict_residuals(d)
                    assert np.allclose(fit, np.load(RESULTS/s/'fc_fd03.npy'), atol=1e-12)
                rng = np.random.default_rng(RNG_SEED+100+10*p+t)
                saved = []
                for f in range(F):
                    idx = block_indices(keep, block, rng); saved.append(idx)
                    fc[p, t, f] = np.corrcoef(residuals[idx], rowvar=False)[I, J]
                    records.append(dict(protocol=protocol, session=s, draw=f,
                        retained=len(idx), unique_frames=len(np.unique(idx)),
                        design_rank=np.nan, residual_df=np.nan,
                        mean_fd=float(d['fd'][idx].mean()), nuisance='frozen_fit'))
                indices[f'{protocol}_{s}'] = np.asarray(saved)
        fc.flush()
    np.savez_compressed(OUT/'block_indices.npz', **indices)
    pd.DataFrame(records).to_csv(OUT/'fc_qc.csv', index=False)
    # Copy only masks (not raw images) for direct inspection alongside joint outputs.
    np.savez_compressed(OUT/'matched_masks.npz', **{p:masks[p] for p in PROTOCOLS})
    return protocols, fc


def strict_residuals(d):
    from .nuisance import pca_signals
    from .connectivity import clean_timeseries
    keep = d['strict']
    pcs = pca_signals(d['tissue'].T, np.ones(d['tissue'].shape[1], bool), keep, True, n=5)[0]
    fit, residuals, _, _, _ = clean_timeseries(d['y'], d['motion'], pcs, keep_override=keep)
    return fit, residuals


def spread(x):
    q = np.quantile(x, [.025, .5, .975])
    return {'q025': q[0], 'median': q[1], 'q975': q[2], 'sd': float(np.std(x)),
            'positive_fraction': float(np.mean(x > 0))}


def analyze(force=False):
    OUT.mkdir(exist_ok=True)
    inputs = lineage()
    design_path = OUT/'design.json'
    if not force and design_path.exists():
        old = json.loads(design_path.read_text())
        if old.get('version') == VERSION and old.get('inputs') == inputs:
            if all(sha(OUT/p) == h for p,h in old['outputs'].items()):
                print('Joint sampling checkpoint verified', flush=True); return
    started = time.time()
    counts = np.asarray([np.load(RESULTS/'sc_reliability'/s/'counts.npy')[:, I, J] for s in SESSIONS])
    # Independent RNG streams per session; bootstrap the 20-run estimator, not one run.
    idx = np.asarray([np.random.default_rng(RNG_SEED+t).integers(20, size=(B,20)) for t in range(2)])
    w, selected, hits = sc_bootstrap(counts, idx)
    np.save(OUT/'sc_run_indices.npy', idx)
    np.save(OUT/'sc_weights.npy', w)
    np.save(OUT/'sc_selected_support.npy', selected)
    fixed = np.load(PRIMARY/'preop/common_support.npy')[I,J]
    point = np.asarray([np.load(PRIMARY/s/'common_per_10000_seeds.npy')[I,J] for s in SESSIONS])
    assert fixed.sum() == 823 and (w[:,:,fixed] > 0).all()
    qc = pd.DataFrame({'draw':np.arange(B), 'common_edges':selected.sum(axis=1),
        'added_edges':(selected & ~fixed).sum(axis=1), 'removed_edges':(~selected & fixed).sum(axis=1),
        'jaccard':(selected & fixed).sum(axis=1)/(selected | fixed).sum(axis=1),
        'preop_own_edges':(hits[0]>=16).sum(axis=1), 'postop_own_edges':(hits[1]>=16).sum(axis=1)})
    qc.to_csv(OUT/'support_qc.csv',index=False)
    pd.DataFrame({'node_i':I+1,'node_j':J+1,'point_common':fixed,
        'bootstrap_common_frequency':selected.mean(axis=0)}).to_csv(OUT/'edge_selection.csv',index=False)
    rows, session_rows, convergence = [], [], []
    with threadpool_limits(limits=1):
        protocols, fc = fc_draws()
        for p, protocol in enumerate(protocols):
            observed = np.asarray([np.load(RESULTS/s/('fc_fd03.npy' if protocol.startswith('strict') else 'fc_primary.npy'))[I,J] for s in SESSIONS])
            for mode in SUPPORTS:
                print('Crossing:', protocol, mode, flush=True)
                mask = fixed if mode == 'fixed' else selected
                rho = evaluate(w, mask, fc[p])
                delta = rho[1]-rho[0]
                # SC-only retains observed full-run FC. FC-only retains point SC/support.
                sc_rho = evaluate(w, mask, observed[:,None,:])[:,:,0]
                fc_rho = evaluate(point[:,None,:], fixed, fc[p])[:,0,:]
                point_rho = evaluate(point[:,None,:], fixed, observed[:,None,:])[:,0,0]
                sc_delta, fc_delta = sc_rho[1]-sc_rho[0], fc_rho[1]-fc_rho[0]
                np.savez_compressed(OUT/f'{protocol}_{mode}.npz', rho=rho, delta=delta,
                    sc_only_rho=sc_rho, fc_only_rho=fc_rho, observed_rho=point_rho)
                base = dict(protocol=protocol, support=mode, SC_draws=B, FC_draws=F,
                    observed_delta=point_rho[1]-point_rho[0],
                    nuisance='fully_refit' if protocol in PROTOCOLS else 'frozen_fit')
                for source, values in [('SC_only', sc_delta), ('FC_only', fc_delta), ('joint',delta)]:
                    rows.append(base | {'source':source, **spread(values), **(variance_components(delta) if source=='joint' else {})})
                for t, session in enumerate(SESSIONS):
                    for source, values in [('SC_only',sc_rho[t]),('FC_only',fc_rho[t]),('joint',rho[t])]:
                        session_rows.append(base | {'session':session,'source':source,
                            'observed_rho':point_rho[t],**spread(values)})
                for axis, half in [('SC_first',delta[:B//2]),('SC_second',delta[B//2:]),
                                    ('FC_first',delta[:,:F//2]),('FC_second',delta[:,F//2:])]:
                    convergence.append(base | {'half':axis,**spread(half)})
    pd.DataFrame(rows).to_csv(OUT/'summary.csv',index=False)
    pd.DataFrame(session_rows).to_csv(OUT/'session_coupling.csv',index=False)
    pd.DataFrame(convergence).to_csv(OUT/'monte_carlo_halves.csv',index=False)
    save_figures()
    outputs = {str(p.relative_to(OUT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['design.json','run.log']}
    design_path.write_text(json.dumps(dict(version=VERSION,inputs=inputs,outputs=outputs,
        SC_draws=B,FC_draws=F,seed=RNG_SEED,protocols=protocols,
        cells_per_crossed_grid=B*F,independent_grid_cells=False,
        interpretation='Conditional resampling ranges, not clinical confidence intervals or probabilities of surgery effects',
        elapsed_seconds=time.time()-started),indent=2))
    print('Joint sampling completed in',round(time.time()-started,1),'seconds',flush=True)


def save_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = pd.read_csv(OUT/'summary.csv')
    protocols = ['primary_block10','strict_block10']+list(PROTOCOLS)
    labels = ['Full primary: blocks (fit fixed)','Full strict: blocks (fit fixed)',
              'Primary: duration','Primary: duration + FD','Primary: duration + FD + half',
              'Strict: duration','Strict: duration + FD','Strict: duration + FD + half']
    fig, ax = plt.subplots(1,2,figsize=(12,5),sharey=True,layout='constrained')
    for a,mode in zip(ax,SUPPORTS):
        for offset,source,color in [(-.18,'SC_only','#509c83'),(0,'FC_only','#c88b39'),(.18,'joint','#346d9d')]:
            v=d[(d.support==mode)&(d.source==source)].set_index('protocol').loc[protocols]
            y=np.arange(len(v))+offset
            a.errorbar(v['median'],y,xerr=np.array([v['median']-v.q025,v.q975-v['median']]),fmt='o',ms=3,capsize=2,color=color,label=source.replace('_',' '))
        a.axvline(0,color='gray',ls='--',lw=1);a.set_xlabel('Post − pre SC–FC coupling Δ')
        a.set_title('Fixed 823-edge support' if mode=='fixed' else '16/20 support reselected')
        a.set_yticks(range(len(labels)),labels,fontsize=9);a.grid(axis='x',alpha=.2)
    ax[0].invert_yaxis()
    ax[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Median and conditional 2.5–97.5% range; dependent crossed cells, not clinical CIs',fontsize=11)
    fig.savefig(RESULTS/'figures/joint_sampling_ranges.png',dpi=180);plt.close(fig)
    v=d[(d.source=='joint')&(d.support=='fixed')].set_index('protocol').loc[protocols]
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    left=np.zeros(len(v))
    for col,label,color in [('SC_share','SC main effect','#509c83'),('FC_share','FC main effect','#c88b39'),('interaction_share','Interaction','#346d9d')]:
        axs[0].barh(range(len(v)),v[col]*100,left=left,color=color,label=label);left+=v[col].to_numpy()*100
    axs[0].set_yticks(range(len(v)),labels,fontsize=8);axs[0].invert_yaxis()
    axs[0].set_xlabel('Share of finite crossed-grid variance (%)')
    axs[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3)
    qc=pd.read_csv(OUT/'support_qc.csv')
    axs[1].hist(qc.common_edges,bins=24,color='#346d9d',alpha=.8)
    axs[1].axvline(823,color='#c88b39',ls='--',label='Observed 823')
    axs[1].set_xlabel('Common edges after support reselection');axs[1].set_ylabel('SC bootstrap draws (500)')
    axs[1].legend(fontsize=8);axs[1].set_title('Selection stability, conditional on 20 runs')
    fig.savefig(RESULTS/'figures/joint_sampling_components.png',dpi=180);plt.close(fig)

