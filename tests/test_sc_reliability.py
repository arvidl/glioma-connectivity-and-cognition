"""Real repeated tracking lineage, threshold estimands and held-fixed FC controls."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pat16 import ROOT,RESULTS,SESSIONS
from pat16.bold import sha
from pat16.tract_repeats import OUT,REPEATS,SEEDS
from pat16.sc_reliability import load,threshold_weights,consensus_weights,THRESHOLDS


def test_full_precision_seed160_replays_reproduce_original_both_sessions():
    for s in SESSIONS:
        p=OUT/s;q=json.loads((p/'baseline_replay.json').read_text())
        assert q['exact_count_matrix_reproduction'] and q['max_count_difference']==0
        assert q['length_filtered_streamlines']==q['original_length_filtered_streamlines']
        assert np.array_equal(np.load(p/'baseline_replay_counts.npy'),np.load(RESULTS/s/'sc_counts.npy'))
        assert np.load(p/'csd_sh_coeff64.npy',mmap_mode='r').dtype==np.float64


def test_all_40_real_tracking_runs_reconstruct_counts_from_saved_endpoints():
    random_seeds=[]
    for s in SESSIONS:
        p=OUT/s;stack=np.load(p/'counts.npy');spec=json.loads((p/'complete.json').read_text())
        assert stack.shape==(REPEATS,100,100) and np.issubdtype(stack.dtype,np.integer)
        assert not np.any(np.diagonal(stack,axis1=1,axis2=2)) and np.array_equal(stack,stack.transpose(0,2,1))
        for r in range(REPEATS):
            q=json.loads((p/f'repeat_{r:02d}_qc.json').read_text());ep=np.load(p/f'repeat_{r:02d}_endpoints.npz')
            random_seeds.append(q['random_seed']);assert q['seeds']==SEEDS==len(ep['seeds_ras_mm'])
            assert q['random_seed']==spec['independent_random_seeds'][r]
            assert np.isfinite(ep['seeds_ras_mm']).all()
            assert np.all((ep['lengths_mm']>=20)&(ep['lengths_mm']<=250))
            pairs=ep['pairs'];valid=pairs[:,0]>=0;ij=pairs[valid]
            assert (ij[:,0]!=ij[:,1]).all() and (ep['distances_mm'][valid]<=6).all()
            c=np.zeros((100,100),int);np.add.at(c,(ij[:,0],ij[:,1]),1);c+=c.T
            assert np.array_equal(c,stack[r])
            assert valid.sum()==q['assigned_interparcel_streamlines']==c.sum()//2
            assert q['counts_sha256']==sha(p/f'repeat_{r:02d}_counts.npy')
        assert not np.array_equal(stack[0],stack[1])
    assert len(set(random_seeds))==2*REPEATS and 160 not in random_seeds


def test_inclusive_threshold_nesting_consensus_frequency_and_original_groups():
    i,j,b,r,fc=load();e=pd.read_csv(OUT/'edges.csv')
    assert len(e)==4950 and np.array_equal(e.node_i,i+1) and np.array_equal(e.node_j,j+1)
    assert np.array_equal(e.baseline_group=='shared',(b>0).all(0))
    for t,s in enumerate(SESSIONS):
        for k in THRESHOLDS:assert np.allclose(e[f'{s}_frequency_ge{k}'],(r[t]>=k).mean(0))
        assert np.allclose(e[s+'_repeat_mean_count'],r[t].mean(0))
        assert np.allclose(e[s+'_repeat_sd_count'],r[t].std(0,ddof=1))
    for k in THRESHOLDS:
        own=threshold_weights(b,k,'own');common=threshold_weights(b,k,'common')
        assert np.array_equal(own>0,b>=k) and np.array_equal(common[0]>0,common[1]>0)
        assert np.array_equal(common>0,np.broadcast_to((b>=k).all(0),b.shape))
        for support in ['own','common']:
            w=consensus_weights(r,k,.8,support);p=(r>=k).mean(1)>=.8
            if support=='common':p[:]=p.all(0)
            assert np.array_equal(w>0,p) and np.allclose(w[p],r.mean(1)[p])
            # A count cutoff applies to runs, not their pooled 600k-seed sum.
            assert np.all((r>=k).mean(1)[w>0]>=.8)


def test_fixed_and_repeat_couplings_match_independent_scipy_and_common_support():
    i,j,b,r,fc=load();fixed=pd.read_csv(OUT/'fixed_coupling.csv')
    for row in fixed.itertuples():
        w=threshold_weights(b,row.threshold,row.support) if row.method=='baseline' else consensus_weights(r,row.threshold,row.consensus_fraction,row.support)
        for stage in ['primary','strict']:
            vals=[]
            for t in range(2):
                k=w[t]>0;vals.append(spearmanr(w[t,k],fc[stage][t,k]).statistic)
            assert np.isclose(getattr(row,stage+'_delta'),vals[1]-vals[0],atol=1e-13)
        if row.support=='common':assert row.pre_edges==row.post_edges==row.shared_edges
    pairs=pd.read_csv(OUT/'repeat_pair_coupling.csv')
    assert len(pairs)==len(THRESHOLDS)*2*REPEATS**2 and not pairs[['threshold','support','pre_repeat','post_repeat']].duplicated().any()
    for row in pairs[(pairs.pre_repeat==0)&(pairs.post_repeat==19)].itertuples():
        w=threshold_weights(np.stack([r[0,0],r[1,19]]),row.threshold,row.support)
        for stage in fc:
            v=[spearmanr(a[a>0],f[a>0]).statistic for a,f in zip(w,fc[stage])]
            assert np.isclose(getattr(row,stage+'_delta'),v[1]-v[0],atol=1e-13)


def test_all_reliability_partitions_exactly_reconstruct_reversal_terms():
    fixed=pd.read_csv(OUT/'fixed_coupling.csv').fillna({'consensus_fraction':-1})
    c=pd.read_csv(OUT/'coupling_contributions.csv').fillna({'consensus_fraction':-1})
    keys=['method','threshold','consensus_fraction','support']
    for partition,g in c.groupby('partition'):
        sums=g.groupby(keys).reversal_term.sum().reset_index().merge(fixed,on=keys)
        assert len(sums)==len(fixed) and np.allclose(sums.reversal_term,sums.reversal_shift,atol=1e-13)
    old=pd.read_csv(RESULTS/'coupling_edge_drivers/edges.csv').reversal_shift.sum()
    assert np.isclose(pd.read_csv(OUT/'original_reversal_by_reliability.csv').original_reversal_term.sum(),old,atol=1e-13)


def test_all_previous_matched_full_refits_reproduced_and_sc_settings_fixed():
    m=pd.read_csv(OUT/'matched_coupling.csv');old=pd.read_csv(RESULTS/'censoring_investigation/matched_replicates.csv');old=old[old['mode']=='refit']
    assert len(m)==6*200*5
    ref=m[m.sc_reference=='baseline_k1']
    for t,s in enumerate(SESSIONS):
        a=ref.merge(old[old.session==s],on=['protocol','replicate'])
        assert len(a)==1200 and np.allclose(a['pre_coupling' if t==0 else 'post_coupling'],a.coupling,atol=1e-9)
    for name,g in m.groupby('sc_reference'):
        assert g.pre_edges.nunique()==1 and g.post_edges.nunique()==1


def test_all_original_derived_inputs_and_previous_analysis_archives_unchanged():
    h=json.loads((OUT/'before_reliability/unchanged_inputs.json').read_text())
    for name,expected in h.items():assert sha((RESULTS/'before_primary_sc'/name) if (RESULTS/'before_primary_sc'/name).exists() else ROOT/name)==expected,name
    sep=r'\clearpage\appendix'
    assert (ROOT/'report/main.tex').read_text().split(sep)[1]==(OUT/'before_reliability/main.tex').read_text().split(sep)[1]
