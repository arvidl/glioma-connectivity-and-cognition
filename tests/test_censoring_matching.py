"""Scientific contracts for paired conditional frame-selection experiments."""
import json
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits
from pat16 import ROOT, RESULTS
from pat16.bold import sha
from pat16.censoring import (OUT, SESSIONS, PROTOCOLS, N, CALIPER, load_session,
                            fit_selected, random_pairs, coupling_measures)
from pat16.connectivity import clean_timeseries


def test_primary_outputs_are_unchanged_by_censoring_investigation():
    hashes=json.loads((OUT/'before_matching/unchanged_primary.json').read_text())
    for name,expected in hashes.items():
        assert sha((RESULTS/'before_primary_sc'/name) if (RESULTS/'before_primary_sc'/name).exists() else ROOT/name)==expected,name


@pytest.mark.parametrize('session',SESSIONS)
def test_cached_tissue_refits_reproduce_complete_primary_and_strict_pipelines(session):
    d=load_session(session)
    with threadpool_limits(limits=1):
        for pool,name in [('primary','primary'),('strict','fd03')]:
            fc,q,_=fit_selected(d,d[pool],'refit')
            assert np.allclose(fc,np.load(RESULTS/session/f'fc_{name}.npy'),atol=1e-8)
            assert q['residual_df']==d[pool].sum()-q['design_rank']
        # Outside-pool data must not enter retained-frame PCA or regression.
        k=d['strict'];changed=dict(d)
        changed['tissue']=d['tissue'].copy();changed['tissue'][~k]+=10000
        changed['y']=d['y'].copy();changed['y'][~k]+=10000
        fc,_,_=fit_selected(changed,k,'refit')
        assert np.allclose(fc,np.load(RESULTS/session/'fc_fd03.npy'),atol=1e-8)


def test_every_saved_draw_obeys_pool_duration_motion_and_temporal_constraints():
    arc=np.load(OUT/'matched_masks.npz');ds=[load_session(s) for s in SESSIONS]
    spec=json.loads((OUT/'design.json').read_text())
    assert spec['repeats']==200 and spec['fd_caliper_mm']==CALIPER
    for protocol,(pool,kind) in PROTOCOLS.items():
        masks=arc[protocol]
        assert masks.shape==(200,2,180) and masks.dtype==bool
        assert (masks.sum(-1)==N).all()
        for i,d in enumerate(ds):assert not (masks[:,i] & ~d[pool]).any()
        if kind=='duration':continue
        for ks in masks:
            groups=(np.arange(180)<90,np.arange(180)>=90) if kind=='motion_time' else (np.ones(180,bool),)
            for g in groups:
                a=np.sort(ds[0]['fd'][ks[0]&g]);b=np.sort(ds[1]['fd'][ks[1]&g])
                assert len(a)==len(b)
                assert np.max(abs(a-b),initial=0)<=CALIPER+1e-12
    r=pd.read_csv(OUT/'matched_replicates.csv')
    q=r[r['mode']=='refit']
    assert (q.design_rank==39).all() and (q.residual_df==29).all()
    assert (r.frames==68).all() and np.isfinite(r.coupling).all()


def test_selected_draw_and_support_control_match_saved_statistics():
    protocol='strict_motion_time';rep=17
    ks=np.load(OUT/'matched_masks.npz')[protocol][rep]
    ds=[load_session(s) for s in SESSIONS]
    common=(ds[0]['sc']>0)&(ds[1]['sc']>0)
    saved=pd.read_csv(OUT/'matched_replicates.csv')
    with threadpool_limits(limits=1):
        for d,k,s in zip(ds,ks,SESSIONS):
            for mode in ('frozen','refit'):
                fc,_,_=fit_selected(d,k,mode)
                values=coupling_measures(fc,d['sc'],common,ds[0]['sc'])
                r=saved[(saved.protocol==protocol)&(saved.replicate==rep)&(saved['mode']==mode)&(saved.session==s)].iloc[0]
                for key,value in values.items():assert np.isclose(r[key],value,atol=1e-10)


def test_matching_reproducibility_and_infeasible_guard():
    ds=[load_session(s) for s in SESSIONS]
    args=[d['fd'] for d in ds]+[d['strict'] for d in ds]
    a,g=random_pairs(*args,np.random.default_rng(41),time_halves=True)
    b,h=random_pairs(*args,np.random.default_rng(41),time_halves=True)
    assert np.array_equal(a,b) and g==h and g<=CALIPER
    with pytest.raises(RuntimeError,match='valid pairs'):
        random_pairs(*args,np.random.default_rng(41),caliper=0)


def test_explicit_mask_preserves_original_time_axis_and_rejects_invalid_masks():
    d=load_session('preop');k=d['strict']
    with threadpool_limits(limits=1):
        normal=clean_timeseries(d['y'],d['motion'],d['pcs'],.3)
        explicit=clean_timeseries(d['y'],d['motion'],d['pcs'],keep_override=k)
    assert np.array_equal(explicit[2],k) and explicit[1].shape==(180,100)
    assert np.allclose(normal[0],explicit[0])
    for bad in [k.astype(int),k[:-1],np.ones(180,bool)]:
        with pytest.raises(ValueError,match='Explicit frame mask'):
            clean_timeseries(d['y'],d['motion'],d['pcs'],keep_override=bad)
