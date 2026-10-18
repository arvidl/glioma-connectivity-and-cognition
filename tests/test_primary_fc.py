"""Primary equal-duration design, saved-estimator replay and preservation contracts."""
import hashlib,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from pat16 import ROOT,RESULTS,SESSIONS
from pat16.bold import sha
from pat16.primary_fc import OUT,PRIMARY,PROTOCOLS,targets,inputs
from pat16.censoring import load_session
from pat16.pc_sampling import tissue_preprocess,loadings
from pat16.joint_refit import refit
from pat16.joint_sampling import I,J


def test_primary_policy_provenance_and_common_duration():
    assert targets([146,150])==dict(half=73,two_thirds=97,four_fifths=116)
    design=json.loads((OUT/'design.json').read_text());assert design['inputs']==inputs()
    for p,h in design['outputs'].items():assert sha(OUT/p)==h,p
    policy=json.loads((OUT/'policy.json').read_text());assert policy['primary_distinct_frames']==97
    assert policy['selection_uses_new_contrast_or_clinical_sign'] is False
    w=np.load(OUT/'frame_weights.npy');tiles=np.load(OUT/'tile_draws.npz')
    for p,protocol in enumerate(PROTOCOLS):
        fraction,block=protocol.rsplit('_block',1);block=int(block);target=policy['targets'][fraction]
        assert w[p].max()==1 and (w[p].sum(-1)==target).all()
        for t,s in enumerate(SESSIONS):
            keep=load_session(s)['primary'];assert not w[p,t,:,~keep].any()
            for draw in [0,17,199]:
                origin=tiles[protocol+'_'+s+'_origins'][draw];order=tiles[protocol+'_'+s+'_permutations'][draw]
                idx=((origin+np.arange(180))%180).reshape(-1,block)[order].ravel();idx=idx[keep[idx]][:target]
                assert np.array_equal(w[p,t,draw],np.bincount(idx,minlength=180))
    qc=pd.read_csv(OUT/'fit_metrics.csv').query("protocol=='two_thirds_block10' and mode=='full_refit'")
    assert len(qc)==400 and qc.eligible.all() and (qc.design_rank==39).all()
    assert (qc.distinct_rows_minus_rank==58).all()


def test_primary_full_refit_and_spearman_replay():
    p=PROTOCOLS.index(PRIMARY);w=np.load(OUT/'frame_weights.npy');fc=np.load(OUT/'fc_draws.npy',mmap_mode='r')
    with threadpool_limits(limits=1):
        for t,s in enumerate(SESSIONS):
            d=load_session(s);idx=np.flatnonzero(w[p,t,17]);z,_,_=tissue_preprocess(d['tissue'],w[p,t,17]);pcs,_,_=loadings(z,w[p,t,17])
            actual=refit(d,idx,pcs)[0][I,J];assert np.allclose(actual,fc[p,2,t,17],atol=1e-10)
    grid=np.load(OUT/f'{PRIMARY}_fixed_full_refit.npz');sc=np.load(RESULTS/'joint_sampling/sc_weights.npy');mask=np.load(RESULTS/'primary_sc/preop/common_support.npy')[I,J]
    for t in range(2):
        assert np.isclose(spearmanr(sc[t,13,mask],fc[p,2,t,17,mask]).statistic,grid['rho'][t,13,17])
    summary=pd.read_csv(OUT/'summary.csv');assert (summary.role=='primary').sum()==1
    for row in summary.itertuples():
        x=np.load(OUT/f'{row.protocol}_{row.support}_{row.mode}.npz')
        assert x['delta'].shape==(500,200) and np.allclose(x['rho'][1]-x['rho'][0],x['delta'])
        assert np.allclose(np.quantile(x['delta'],[.025,.5,.975]),[row.q025,row.median,row.q975])
        assert np.isclose(row.SC_variance+row.FC_variance+row.interaction_variance,row.total_variance)
    point=json.loads((OUT/'policy.json').read_text())['observed_coupling']
    old=json.loads((RESULTS/'summary.json').read_text())
    for s in SESSIONS:assert np.isclose(point[s],old['sessions'][s]['sc_fc_spearman'])


def test_primary_network_and_clinical_integration_preserves_sources():
    old=pd.read_csv(RESULTS/'evidence_integration/assessment_evidence_ledger.csv');new=pd.read_csv(OUT/'clinical_evidence_ledger.csv')
    pd.testing.assert_frame_equal(old,new[old.columns]);assert len(new)==47
    net=pd.read_csv(OUT/'network_uncertainty.csv').query("role=='primary'");assert len(net)==14
    draws=np.load(OUT/'primary_network_draws.npz')
    for row in net.itertuples():
        d=draws[row.network+'_'+row.modality]
        assert np.allclose(np.quantile(d[1]-d[0],[.025,.5,.975]),[row.q025,row.median,row.q975])


def test_preceding_delivery_and_complete_clinical_appendices_preserved():
    base=RESULTS/'before_primary_fc';manifest=json.loads((base/'manifest.json').read_text())
    for row in manifest['files']:
        p=(ROOT if row['immutable'] else base)/row['path']
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],row['path']
    appendix=(ROOT/'report/main.tex').read_text().split('\\section{Complete supplied assessments}',1)[1]
    assert hashlib.sha256(('\\section{Complete supplied assessments}'+appendix).encode()).hexdigest()==manifest['clinical_appendix_source_sha256']
    for name in ['cantab_all','cantab_all_dictionary','psychosocial_all','psychosocial_all_dictionary']:
        assert sha(ROOT/'report/generated'/f'{name}.tex')==sha(base/'report/generated'/f'{name}.tex')
