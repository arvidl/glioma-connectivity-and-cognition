"""Frequency weighting, paired replay, fit feasibility and clinical traceability."""
import hashlib,json
import numpy as np
import pandas as pd
from scipy.linalg import lstsq,subspace_angles
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from pat16 import ROOT,RESULTS
from pat16.bold import sha
from pat16.censoring import load_session
from pat16.nuisance import pca_signals
from pat16.joint_refit import (OUT,PROTOCOLS,MODES,multiplicities,weighted_pcs,regression_design,refit,inputs)
from pat16.joint_sampling import OUT as JOINT,I,J
from pat16.evidence_integration import OUT as EVIDENCE


def test_weighted_pca_matches_explicit_duplicate_estimator():
    with threadpool_limits(limits=1):
        d=load_session('preop');idx=np.random.default_rng(83).choice(np.flatnonzero(d['primary']),146)
        a,var,n=weighted_pcs(d['tissue'],idx)
        b,_=pca_signals(d['tissue'].T,np.ones(d['tissue'].shape[1],bool),idx,True,n=5)
        assert subspace_angles(a[idx],b[idx]).max()<1e-10
        assert n==d['tissue'].shape[1] and np.isfinite(var).all()


def test_refit_reproduces_full_run_fc_and_explicit_duplicate_wls():
    with threadpool_limits(limits=1):
        for s in ('preop','postop'):
            d=load_session(s)
            for pool,name in [('primary','fc_primary.npy'),('strict','fc_fd03.npy')]:
                idx=np.flatnonzero(d[pool]);fc,q,_,_=refit(d,idx)
                assert q['eligible'] and q['design_rank']==39
                assert np.allclose(fc,np.load(RESULTS/s/name),atol=1e-11)
        idx=np.flatnonzero(d['primary'])
        idx=np.random.default_rng(12).choice(idx,len(idx))
        fc,q,_,res=refit(d,idx,d['pcs'])
        w=multiplicities(idx,180);x=regression_design(d['motion'],d['pcs'],w)
        beta=lstsq(x[idx],d['y'][idx],cond=1e-8)[0];expected=d['y']-x@beta
        assert np.allclose(expected,res,atol=1e-10)
        c=np.corrcoef(expected[idx].T);np.fill_diagonal(c,0)
        assert np.allclose(c,fc,atol=1e-11)
        # Derivative at frame 70 comes from original frame 69, not an artificial join.
        idx2=idx[::-1];assert np.array_equal(regression_design(d['motion'],d['pcs'],multiplicities(idx2,180)),x)


def test_repeated_frames_do_not_rescue_sparse_fits():
    with threadpool_limits(limits=1):
        d=load_session('preop');idx=np.tile(np.flatnonzero(d['primary'])[:40],4)
        fc,q,_,_=refit(d,idx,d['pcs'])
        assert q['sample_rows']==160 and q['unique_frames']==40
        assert not q['eligible'] and np.isnan(fc).all()
        assert 'distinct' in q['reason']


def test_actual_paired_models_replay_frozen_and_crossed_statistics():
    d=json.loads((OUT/'design.json').read_text());assert d['inputs']==inputs()
    for p,h in d['outputs'].items():assert sha(OUT/p)==h,p
    old=json.loads((JOINT/'design.json').read_text())
    f=np.load(OUT/'fc_draws.npy',mmap_mode='r');of=np.load(JOINT/'fc_draws.npy',mmap_mode='r')
    for p,protocol in enumerate(PROTOCOLS):
        assert np.array_equal(f[p,0],of[old['protocols'].index(protocol)])
        ix=np.load(OUT/f'{protocol}_paired_draws.npy')
        assert np.array_equal(ix,np.flatnonzero(np.isfinite(f[p,1:]).all(axis=(0,1,3))))
    summary=pd.read_csv(OUT/'summary.csv');weights=np.load(JOINT/'sc_weights.npy')
    fixed=np.load(RESULTS/'primary_sc/preop/common_support.npy')[I,J]
    for r in summary.itertuples():
        z=np.load(OUT/f'{r.protocol}_{r.support}_{r.mode}.npz');delta=z['delta']
        assert np.allclose(delta,z['rho'][1]-z['rho'][0])
        assert np.allclose(np.quantile(delta,[.025,.5,.975]),[r.q025,r.median,r.q975])
        assert np.isclose(r.SC_share+r.FC_share+r.interaction_share,1)
        b=17;k=39;t=1;mask=fixed if r.support=='fixed' else np.load(JOINT/'sc_selected_support.npy')[b]
        v=f[PROTOCOLS.index(r.protocol),MODES.index(r.mode),t,z['FC_draw_ids'][k]]
        assert np.isclose(z['rho'][t,b,k],spearmanr(weights[t,b,mask],v[mask]).statistic,atol=1e-13)
    for protocol in ('primary_block5','primary_block10','primary_block20'):
        for support in ('fixed','reselected'):
            shift=np.load(OUT/f'{protocol}_{support}_paired_shifts.npz')
            assert np.allclose(shift['coefficients_minus_frozen']+shift['full_minus_coefficients'],shift['full_minus_frozen'])
    feasible=pd.read_csv(OUT/'feasibility.csv')
    for r in feasible.itertuples():
        assert bool(r.summary_available)==(r.paired_eligible>=100)
        if not r.summary_available:assert r.protocol not in summary.protocol.values


def test_all_clinical_fields_and_network_points_trace_to_sources():
    a=pd.read_csv(RESULTS/'assessments_all.csv');l=pd.read_csv(EVIDENCE/'assessment_evidence_ledger.csv')
    pd.testing.assert_frame_equal(l[a.columns],a)
    assert len(l)==47 and (l.group=='CANTAB').sum()==28
    assert l[l.group!='CANTAB'].candidate_networks.isna().all()
    assert l[l.measure=='sport_freq_per_week'].availability.iloc[0]=='pre-only'
    assert l[l.measure=='employment'].postop.iloc[0]=='Sickleave'
    n=pd.read_csv(EVIDENCE/'network_uncertainty.csv');source=pd.read_csv(RESULTS/'network_summary.csv')
    for r in n.itertuples():
        for s in ('preop','postop'):
            v=source[(source.session==s)&(source.network==r.network)].iloc[0]
            expected=v.within_sc_weight if r.modality=='SC' else v.within_fc_fisher_z
            assert np.isclose(getattr(r,s),expected,atol=1e-12)
    d=json.loads((EVIDENCE/'design.json').read_text())
    for p,h in d['inputs'].items():assert sha(ROOT/p)==h,p
    for p,h in d['outputs'].items():assert sha(EVIDENCE/p)==h,p


def test_preceding_outputs_and_clinical_appendices_unchanged():
    archive=RESULTS/'before_joint_refit';m=json.loads((archive/'manifest.json').read_text())
    for r in m['files']:
        p=ROOT/r['path'] if r['immutable'] else archive/r['path']
        assert sha(p)==r['sha256'],p
    source='\\section{Complete supplied assessments}'+(ROOT/'report/main.tex').read_text().split('\\section{Complete supplied assessments}')[1]
    assert hashlib.sha256(source.encode()).hexdigest()==m['clinical_appendix_source_sha256']
