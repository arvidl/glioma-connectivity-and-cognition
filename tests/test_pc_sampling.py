"""Nonoverlapping original-time sampling, PC-span algebra and saved evidence."""
import json,hashlib
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from threadpoolctl import threadpool_limits
from pat16 import ROOT,RESULTS
from pat16.bold import sha
from pat16.censoring import load_session
from pat16.joint_refit import weighted_pcs,refit,multiplicities
from pat16.pc_sampling import (OUT,JOINT,REFIT,PROTOCOLS,DESIGNS,BLOCKS,MODES,block_subset,tissue_preprocess,loadings,background_fit,adjusted_overlap,inputs)


def test_nonoverlapping_tiles_exact_censoring_targets_and_replay():
    keep=np.ones(180,bool);keep[:4]=False;keep[42:58]=False;keep[101:120]=False
    for b in BLOCKS:
        for target in (60,93,keep.sum()):
            a,origin,order=block_subset(keep,b,target,np.random.default_rng(8))
            aa,oo,pp=block_subset(keep,b,target,np.random.default_rng(8))
            assert np.array_equal(a,aa) and origin==oo and np.array_equal(order,pp)
            assert len(a)==len(np.unique(a))==target and keep[a].all()
            tile=((origin+np.arange(180))%180).reshape(-1,b)[order].ravel()
            assert np.array_equal(a,tile[keep[tile]][:target])
            if target==keep.sum():assert np.array_equal(np.sort(a),np.flatnonzero(keep))
    import pytest
    with pytest.raises(ValueError):block_subset(keep,7,90,np.random.default_rng(2))


def test_new_pca_matches_old_and_temporal_only_control_is_invariant():
    with threadpool_limits(limits=1):
        saved=np.load(JOINT/'block_indices.npz')
        for s in ('preop','postop'):
            d=load_session(s);idx=saved['primary_block10_'+s][17];w=multiplicities(idx,180)
            z0,sd0,valid0=tissue_preprocess(d['tissue'],d['primary'].astype(float));_,v0,_=loadings(z0,d['primary'].astype(float))
            z,sd,valid=tissue_preprocess(d['tissue'],w);scores,_,eig=loadings(z,w)
            old,_,_=weighted_pcs(d['tissue'],idx)
            assert subspace_angles(scores[idx],old[idx]).max()<1e-10
            assert np.all(eig[:-1]>=eig[1:])
            temporal,_,_=tissue_preprocess(d['tissue'],w,sd_override=sd0)
            a=refit(d,idx,d['pcs'])[0];b=refit(d,idx,temporal@v0)[0]
            assert np.allclose(a,b,atol=1e-10)
            _,x=background_fit(d,w)
            ov=adjusted_overlap(d['pcs'],temporal@v0,x,w)
            assert ov['adjusted_PC_cos2_min']>1-1e-10
            full_idx=np.flatnonzero(d['primary']);fullw=d['primary'].astype(float)
            fullscores=loadings(z0,fullw)[0]
            actual=refit(d,full_idx,fullscores)[0]
            assert np.allclose(actual,np.load(RESULTS/s/'fc_primary.npy'),atol=1e-10)


def test_actual_sampling_contracts_and_frozen_refit_replay():
    d=json.loads((OUT/'design.json').read_text());assert d['inputs']==inputs()
    for p,h in d['outputs'].items():assert sha(OUT/p)==h,p
    w=np.load(OUT/'frame_weights.npy');fc=np.load(OUT/'fc_draws.npy',mmap_mode='r');old=np.load(REFIT/'fc_draws.npy',mmap_mode='r')
    tiles=np.load(OUT/'tile_draws.npz')
    for b in BLOCKS:
        boot=w[PROTOCOLS.index(f'bootstrap_block{b}')];unit=w[PROTOCOLS.index(f'unit_coverage_block{b}')]
        sampled=w[PROTOCOLS.index(f'block_unique_block{b}')];high=w[PROTOCOLS.index(f'block80_block{b}')]
        assert np.array_equal(unit,boot>0)
        assert np.array_equal(sampled.sum(-1),(boot>0).sum(-1))
        assert np.max(sampled)<=1 and np.max(high)<=1
        assert np.allclose(fc[PROTOCOLS.index(f'bootstrap_block{b}')],old[BLOCKS.index(b)],atol=1e-10)
        for t,s in enumerate(('preop','postop')):
            keep=load_session(s)['primary'];assert not w[:,t,:,~keep].any()
            assert (high[t].sum(-1)==int(.8*keep.sum())).all()
            for design in ['block_unique','block80']:
                protocol=f'{design}_block{b}';prefix=protocol+'_'+s
                origin=tiles[prefix+'_origins'][17];order=tiles[prefix+'_permutations'][17]
                idx=((origin+np.arange(180))%180).reshape(-1,b)[order].ravel();target=w[PROTOCOLS.index(protocol),t,17].sum()
                idx=idx[keep[idx]][:target];expected=np.bincount(idx,minlength=180)
                assert np.array_equal(w[PROTOCOLS.index(protocol),t,17],expected)
    for r in pd.read_csv(OUT/'summary.csv').itertuples():
        z=np.load(OUT/f'{r.protocol}_{r.support}_{r.mode}.npz')
        assert z['delta'].shape==(500,200)
        assert np.allclose(z['delta'],z['rho'][1]-z['rho'][0])
        assert np.allclose(np.quantile(z['delta'],[.025,.5,.975]),[r.q025,r.median,r.q975])
        assert np.isclose(r.SC_share+r.FC_share+r.interaction_share,1)
    # Recompute one actual new draw per design/session from the source time series.
    with threadpool_limits(limits=1):
        for t,s in enumerate(('preop','postop')):
            d=load_session(s)
            for design in ('unit_coverage','block_unique','block80'):
                p=PROTOCOLS.index(design+'_block10');weight=w[p,t,17]
                idx=np.repeat(np.arange(180),weight)
                actual,q,_,_=refit(d,idx)
                assert q['eligible'] and q['design_rank']==39
                i,j=np.triu_indices(100,1)
                assert np.allclose(actual[i,j],fc[p,2,t,17],atol=1e-10)


def test_mechanism_algebra_and_clinical_supplement():
    f=np.load(OUT/'mechanism_fc.npy');p=PROTOCOLS.index('bootstrap_block10');actual=np.load(OUT/'fc_draws.npy',mmap_mode='r')
    assert np.allclose(f[0],actual[p,1],atol=1e-10)
    assert np.allclose(f[-1],actual[p,2],atol=1e-10)
    assert np.allclose(f[0],f[1],atol=1e-10)
    sh=np.load(OUT/'mechanism_shifts.npz')
    assert np.allclose(sh['preprocessing_minus_coefficients']+sh['full_minus_preprocessing'],sh['full_minus_coefficients'])
    assert np.allclose(sh['pca_minus_coefficients']+sh['full_minus_pca'],sh['full_minus_coefficients'])
    assert np.max(np.abs(sh['temporal_minus_coefficients']))<1e-9
    original=pd.read_csv(RESULTS/'evidence_integration/assessment_evidence_ledger.csv');a=pd.read_csv(OUT/'clinical_field_links.csv')
    pd.testing.assert_frame_equal(a[original.columns],original)
    assert len(a)==47 and (a.group=='CANTAB').sum()==28
    assert a[a.group!='CANTAB'].candidate_networks.isna().all()
    ranges=pd.read_csv(OUT/'network_ranges.csv');source=pd.read_csv(RESULTS/'network_summary.csv')
    assert len(ranges)==12*3*7
    for r in ranges.itertuples():
        for s in ('preop','postop'):
            assert np.isclose(getattr(r,s),source[(source.session==s)&(source.network==r.network)].within_fc_fisher_z.iloc[0])


def test_prior_delivery_and_all_clinical_appendices_preserved():
    a=RESULTS/'before_pc_sampling';m=json.loads((a/'manifest.json').read_text())
    for r in m['files']:
        p=ROOT/r['path'] if r['immutable'] else a/r['path'];assert sha(p)==r['sha256'],p
    text='\\section{Complete supplied assessments}'+(ROOT/'report/main.tex').read_text().split('\\section{Complete supplied assessments}')[1]
    assert hashlib.sha256(text.encode()).hexdigest()==m['clinical_appendix_source_sha256']
