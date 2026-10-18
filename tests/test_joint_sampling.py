"""Resampling units, actual-support ranks and finite crossed-grid identities."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pat16 import ROOT, RESULTS, SESSIONS
from pat16.bold import sha
from pat16.joint_sampling import (OUT, B, F, I, J, sc_bootstrap, block_indices,
    evaluate, variance_components, standardized_ranks)


def test_whole_run_bootstrap_preserves_covariance_zeros_and_exact_threshold():
    c=np.zeros((2,20,5),int)
    c[:,:,0]=np.arange(20);c[:,:,1]=2*c[:,:,0]
    c[:,:16,2]=1;c[:,:15,3]=1;c[:,:,4]=3
    idx=np.tile(np.arange(20),(2,3,1))
    w,s,h=sc_bootstrap(c,idx)
    assert np.array_equal(h[:,0,2],[16,16])
    assert s[0,2] and not s[0,3]
    assert np.allclose(w[:,0],c.mean(axis=1)/3)
    assert np.allclose(w[:,:,1],2*w[:,:,0])
    idx[:,:,0]=19
    wb,sb,hb=sc_bootstrap(c,idx)
    assert not sb[:,2].any()  # replacing run 0 with a miss changes 16 to 15
    assert np.allclose(wb[:,:,1],2*wb[:,:,0])


def test_blocks_sample_original_axis_not_concatenated_good_frames():
    keep=np.ones(180,bool);keep[:4]=False;keep[30:100]=False
    x=block_indices(keep,10,np.random.default_rng(4160))
    assert len(x)==keep.sum() and keep[x].all()
    assert np.array_equal(x,block_indices(keep,10,np.random.default_rng(4160)))
    assert len(np.unique(x)) < len(x)
    # All-present input comes in actual circular contiguous blocks.
    y=block_indices(np.ones(180,bool),10,np.random.default_rng(123))
    assert np.all(np.diff(y.reshape(-1,10),axis=1)%180==1)


def test_vectorized_tied_spearman_and_selection_reranking_match_scipy():
    rng=np.random.default_rng(9)
    w=rng.integers(1,8,(2,4,20)).astype(float)
    fc=rng.normal(size=(2,3,20))
    fixed=np.arange(20)%3!=0
    masks=rng.random((4,20))>.3
    for m in [fixed,masks]:
        actual=evaluate(w,m,fc)
        for t in range(2):
            for b in range(4):
                use=m if m.ndim==1 else m[b]
                for f in range(3):
                    expected=spearmanr(w[t,b,use],fc[t,f,use]).statistic
                    assert np.isclose(actual[t,b,f],expected,atol=1e-14)
    assert np.isclose(np.linalg.norm(standardized_ranks([1,1,2,3])),1)


def test_crossed_variance_not_sum_of_anchor_variances():
    a=np.array([-2.,0,3.])[:,None]; b=np.array([-1.,0,1,2])[None,:]
    x=4+a+2*b+.4*a*b
    v=variance_components(x)
    assert np.isclose(v['SC_variance']+v['FC_variance']+v['interaction_variance'],np.var(x))
    assert np.isclose(v['SC_share']+v['FC_share']+v['interaction_share'],1)
    assert v['interaction_variance']>0


def test_actual_joint_design_and_exact_replay_of_matched_fc():
    d=json.loads((OUT/'design.json').read_text())
    assert (d['SC_draws'],d['FC_draws'])==(B,F)==(500,200)
    assert d['independent_grid_cells'] is False and len(d['protocols'])==12
    for path,h in d['inputs'].items():assert sha(ROOT/path)==h,path
    for path,h in d['outputs'].items():assert sha(OUT/path)==h,path
    fc=np.load(OUT/'fc_draws.npy',mmap_mode='r')
    w=np.load(OUT/'sc_weights.npy',mmap_mode='r')
    original=pd.read_csv(RESULTS/'primary_sc/matched_coupling.csv')
    for p,protocol in enumerate(d['protocols'][:6]):
        z=np.load(OUT/f'{protocol}_fixed.npz')
        previous=original[original.protocol==protocol].sort_values('replicate')
        # Existing table fields are checked explicitly in the delivery audit too.
        delta=z['fc_only_rho'][1]-z['fc_only_rho'][0]
        col='delta' if 'delta' in previous else 'coupling_delta'
        assert np.allclose(delta,previous[col],atol=1e-12)
        if protocol.startswith('strict'):
            assert np.allclose(fc[p,0],fc[p,0,0],atol=1e-12)
        mask=np.load(RESULTS/'primary_sc/preop/common_support.npy')[I,J]
        for t in range(2):
            assert np.isclose(z['rho'][t,17,39],spearmanr(w[t,17,mask],fc[p,t,39,mask]).statistic,atol=1e-13)
    summary=pd.read_csv(OUT/'summary.csv')
    assert len(summary)==72
    sessions=pd.read_csv(OUT/'session_coupling.csv')
    assert len(sessions)==144
    assert np.isfinite(summary[['median','q025','q975','sd']]).all().all()
    joint=summary[summary.source=='joint']
    assert np.allclose(joint.SC_share+joint.FC_share+joint.interaction_share,1)
    for r in joint.itertuples():
        z=np.load(OUT/f'{r.protocol}_{r.support}.npz')
        assert z['delta'].shape==(B,F)
        assert np.allclose(z['delta'],z['rho'][1]-z['rho'][0])
        assert np.isclose(z['delta'].var(),r.total_variance,atol=1e-14)
        assert np.allclose(np.quantile(z['delta'],[.025,.5,.975]),[r.q025,r.median,r.q975])


def test_preceding_primary_delivery_and_imaging_are_unchanged():
    archive=RESULTS/'before_joint_sampling'
    m=json.loads((archive/'manifest.json').read_text())
    for r in m['files']:
        p=ROOT/r['path'] if r['immutable'] else archive/r['path']
        assert sha(p)==r['sha256'],p
