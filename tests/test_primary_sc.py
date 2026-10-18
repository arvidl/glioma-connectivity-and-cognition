"""Check the actual primary SC estimands, migration and retained sensitivities."""
import json
import numpy as np
import pandas as pd
from pat16 import ROOT, RESULTS, SESSIONS
from pat16.bold import sha
from pat16.primary_sc import OUT, load_weights, policy_arrays, VERSION
from pat16.networks import metrics, network_means, sc_fc_coupling, graph_growth


def test_policy_support_and_unconditional_mean_rate_from_all_real_repeats():
    counts=np.array([np.load(RESULTS/'sc_reliability'/s/'counts.npy') for s in SESSIONS])
    mean,hits,own,common=policy_arrays(counts)
    assert np.triu(common,1).sum()==823
    assert [np.triu(x,1).sum() for x in own]==[1160,1163]
    for t,s in enumerate(SESSIONS):
        assert np.array_equal(hits[t],np.load(OUT/s/'detection_hits.npy'))
        assert np.array_equal(own[t],np.load(OUT/s/'own_support.npy'))
        assert np.array_equal(common,np.load(OUT/s/'common_support.npy'))
        for support,mask in [('own',own[t]),('common',common)]:
            w=load_weights(s,support)
            assert np.array_equal(w,np.where(mask,mean[t]/3,0.)) or np.allclose(w,np.where(mask,mean[t]/3,0.),atol=1e-14)
            assert np.array_equal(w,w.T) and not np.diag(w).any()
        # Include selected edges missed in some runs; do not take detection-only mean.
        partial=own[t]&(hits[t]<20)
        assert partial.any() and np.all(mean[t][partial] < counts[t].sum(0)[partial]/hits[t][partial])
    policy=json.loads((OUT/'policy.json').read_text())
    assert policy['selection_uses_FC_or_cognition'] is False
    assert policy['source_sha256']==sha(ROOT/'src/pat16/primary_sc.py')
    assert policy['config_sha256']==sha(ROOT/'config/PRIMARY_SC.md')


def test_primary_summary_and_all_fc_controls_use_fixed_common_consensus():
    summary=json.loads((RESULTS/'summary.json').read_text());sens=pd.read_csv(RESULTS/'sensitivity.csv')
    valid=np.load(RESULTS/'common_fc_nodes.npy')
    assert summary['primary_sc_policy']==VERSION
    for s in SESSIONS:
        w=load_weights(s,'common')[np.ix_(valid,valid)]
        rho=sc_fc_coupling(w,np.load(RESULTS/s/'fc_primary.npy')[np.ix_(valid,valid)])
        assert np.isclose(summary['sessions'][s]['sc_fc_spearman'],rho,atol=1e-12)
        for name in ['primary','fd03','fd10','gsr']:
            row=sens[(sens.session==s)&(sens.analysis==name)].iloc[0]
            assert np.isclose(row.sc_fc_spearman,sc_fc_coupling(w,np.load(RESULTS/s/f'fc_{name}.npy')[np.ix_(valid,valid)]),atol=1e-12)
    fixed=pd.read_csv(OUT/'coupling_sensitivity.csv')
    assert sum(fixed.coupling_role=='primary')==sum(fixed.structural_role=='primary')==1
    p=fixed[fixed.coupling_role=='primary'].iloc[0]
    assert p.support=='common' and p.pre_edges==p.post_edges==823
    assert np.isclose(p.primary_delta,-.04172772719230558)
    assert np.isclose(p.strict_delta,-.03423908266679743)


def test_primary_graphs_networks_and_simulations_use_own_consensus():
    graph=pd.read_csv(RESULTS/'graph_metrics.csv');net=pd.read_csv(RESULTS/'network_summary.csv')
    groups=pd.read_csv(RESULTS/'nodes.csv').network.to_numpy()
    scale=max(load_weights(s).max() for s in SESSIONS)
    for s in SESSIONS:
        w=load_weights(s)
        for _,r in graph[(graph.session==s)&(graph.modality=='SC')].iterrows():
            expected=metrics(w,r.target_density,scale)
            for k,v in expected.items():assert np.isclose(r[k],v,atol=1e-12)
        means,names=network_means(w,groups)
        for i,name in enumerate(names):
            actual=net[(net.session==s)&(net.network==name)].iloc[0]
            assert np.isclose(actual.within_sc_weight,means[i,i])
    nodes=pd.read_csv(RESULTS/'nodes.csv')
    expected=graph_growth(load_weights('preop'),int(np.argmax(nodes.lesion_overlap_fraction)),steps=100,dt=.2,rho=.3)
    assert np.allclose(expected,np.load(RESULTS/'simulation_graph_growth.npy'))
    old=np.load(RESULTS/'before_primary_sc/results/simulation_graph_growth.npy')
    assert not np.allclose(old,expected)


def test_primary_rank_accounting_matching_and_all_sensitivity_designs():
    stages=pd.read_csv(OUT/'censoring_decomposition.csv');edges=pd.read_csv(OUT/'edge_contributions.csv')
    networks=pd.read_csv(OUT/'network_contributions.csv')
    assert len(edges)==823 and networks.edges.sum()==823
    for tag,stage in [('primary','primary'),('strict','pca')]:
        expected=stages[stages.stage==stage].iloc[0].delta
        assert np.isclose(edges[tag+'_delta_term'].sum(),expected,atol=1e-12)
        assert np.isclose(networks[tag+'_delta_term'].sum(),expected,atol=1e-12)
    assert np.isclose(edges.reversal_shift_term.sum(),stages.iloc[-1].delta-stages.iloc[0].delta,atol=1e-12)
    current=pd.read_csv(OUT/'matched_coupling.csv')
    previous=pd.read_csv(RESULTS/'sc_reliability/matched_coupling.csv')
    pd.testing.assert_frame_equal(current,previous[previous.sc_reference=='consensus80_common'].reset_index(drop=True))
    assert len(current)==1200
    sensitivities=pd.read_csv(OUT/'coupling_sensitivity.csv')
    assert {'original_fraction','original_k1','original_k20','repeat_mean_unfiltered','consensus50_k1','consensus95_k1','consensus100_k1','consensus80_k5'}<=set(sensitivities.design)
    # Scalar normalization preserves ranks but need not preserve graph contrasts.
    own=sensitivities[sensitivities.support=='own'].set_index('design')
    assert np.isclose(own.loc['original_fraction','primary_delta'],own.loc['original_k1','primary_delta'])


def test_previous_delivery_archive_and_all_protected_inputs_are_intact():
    archive=RESULTS/'before_primary_sc';manifest=json.loads((archive/'manifest.json').read_text())
    for rec in manifest['files']:
        p=ROOT/rec['path'] if rec['immutable'] else archive/rec['path']
        assert sha(p)==rec['sha256'],str(p)
    clinical=(ROOT/'report/main.tex').read_text().split('\\section{Complete supplied assessments}')[1]
    assert sha_text('\\section{Complete supplied assessments}'+clinical)==manifest['clinical_appendix_source_sha256']


def sha_text(s):
    import hashlib
    return hashlib.sha256(s.encode()).hexdigest()
