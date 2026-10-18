"""Contracts for rank accounting, changed-support diagnostics and matched replay."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pat16 import ROOT,RESULTS
from pat16.bold import sha
from pat16.edge_drivers import OUT,rank_contributions,inputs,STAGES,GROUPS


def test_average_rank_ties_and_absent_edges_match_spearman():
    sc=np.array([0,1,1,2,2,7,10]);fc=np.array([100,.1,.1,-.1,.3,.8,.4])
    c=rank_contributions(sc,fc);k=sc>0
    assert c[0]==0 and np.isclose(c.sum(),spearmanr(sc[k],fc[k]).statistic)
    # Excluded FC values have no influence on ranks or normalization.
    fc[0]=-10000
    assert np.array_equal(c,rank_contributions(sc,fc))
    perm=np.array([3,6,1,2,5,0,4])
    assert np.allclose(c[perm],rank_contributions(sc[perm],fc[perm]))
    assert np.isnan(rank_contributions(np.ones(4),np.arange(4))).all()


def test_edge_union_groups_and_node_labels_match_structural_inputs():
    nodes,e,counts,fc=inputs();stored=pd.read_csv(OUT/'edges.csv')
    assert len(e)==len(stored) and e.node_i.le(e.node_j).all()
    assert not stored[['node_i','node_j']].duplicated().any()
    for t,s in enumerate(['preop','postop']):
        sc=np.load(RESULTS/s/'sc_counts.npy')
        assert np.array_equal(counts[t],sc[stored.node_i-1,stored.node_j-1])
    assert np.array_equal(stored.support_group=='shared',(counts>0).all(0))
    assert np.array_equal(stored.support_group=='pre_only',(counts[0]>0)&(counts[1]==0))
    assert np.array_equal(stored.support_group=='post_only',(counts[1]>0)&(counts[0]==0))
    assert np.array_equal(stored.label_i,nodes.set_index('id').loc[stored.node_i,'label'])
    for stage in STAGES:
        for t,s in enumerate(['preop','postop']):
            if stage in ['primary','pca']:
                name='primary' if stage=='primary' else 'fd03'
                delivered=np.load(RESULTS/s/f'fc_{name}.npy')
                assert np.allclose(fc[stage][t],delivered[stored.node_i-1,stored.node_j-1])


def test_exact_edge_support_and_network_partition_sums():
    _,e,counts,fc=inputs();stored=pd.read_csv(OUT/'edges.csv')
    support=pd.read_csv(OUT/'support_summary.csv');pairs=pd.read_csv(OUT/'network_pairs.csv')
    delta={}
    for stage in STAGES:
        rho=[]
        for t,s in enumerate(['preop','postop']):
            k=counts[t]>0;rho.append(spearmanr(counts[t,k],fc[stage][t,k]).statistic)
            assert np.isclose(stored[f'{s}_{stage}_contribution'].sum(),rho[-1],atol=1e-13)
            assert (stored.loc[~k,f'{s}_{stage}_contribution']==0).all()
        delta[stage]=rho[1]-rho[0]
        for table in [stored,support,pairs]:
            assert np.isclose(table[stage+'_delta_contribution'].sum(),delta[stage],atol=1e-13)
    for table in [stored,support,pairs]:
        assert np.isclose(table.reversal_shift.sum(),delta['pca']-delta['primary'],atol=1e-13)
    assert pairs.edges.sum()==len(e) and support.edges.sum()==len(e)
    assert np.allclose(pairs.reversal_shift,pairs[[g+'_shift' for g in GROUPS]].sum(1))


def test_within_and_between_group_terms_reconstruct_pooled_not_subgroup_rho():
    context=pd.read_csv(OUT/'rank_context.csv')
    assert np.allclose(context.pooled_contribution,context.within_group_global_rank_term+context.between_group_global_rank_term,atol=1e-13)
    edges=pd.read_csv(OUT/'edges.csv')
    for row in context.itertuples():
        k=edges.support_group==row.support_group
        assert np.isclose(row.pooled_contribution,edges.loc[k,f'{row.session}_{row.stage}_contribution'].sum())
    # A single-session subset correlation is not an additive pooled contribution.
    assert not np.allclose(context.subgroup_spearman,context.pooled_contribution)


def test_reranked_group_and_single_edge_deletions_match_independent_scipy():
    _,e,counts,fc=inputs();d=pd.read_csv(OUT/'deletions.csv');stored=pd.read_csv(OUT/'edges.csv')
    masks=[(e.support_group=='pre_only').values,(e.support_group=='post_only').values,(e.support_group!='shared').values]
    rows=[d[(d.kind=='support')&(d.removed==g)].iloc[0] for g in ['pre_only','post_only','all_session_specific']]
    for idx in [0,int(stored.reversal_shift.idxmax()),int(stored.reversal_shift.idxmin())]:
        mask=np.zeros(len(e),bool);mask[idx]=True;masks.append(mask)
        row=stored.iloc[idx]
        rows.append({'primary_delta':row.leave_one_primary_delta,'pca_delta':row.leave_one_pca_delta})
    for mask,row in zip(masks,rows):
        for stage in ['primary','pca']:
            vals=[]
            for t in range(2):
                k=(counts[t]>0)&~mask;vals.append(spearmanr(counts[t,k],fc[stage][t,k]).statistic)
            assert np.isclose(row[stage+'_delta'],vals[1]-vals[0],atol=1e-13)


def test_all_saved_matched_contrasts_and_same_mask_refit_effect_reconstruct_prior_results():
    archive=np.load(OUT/'matched_edge_contrasts.npz');old=pd.read_csv(RESULTS/'censoring_investigation/matched_replicates.csv')
    groups=pd.read_csv(OUT/'matched_group_summary.csv');e=pd.read_csv(OUT/'edges.csv')
    for protocol in old.protocol.unique():
        a=archive[protocol+'__frozen'];b=archive[protocol+'__refit']
        assert a.shape==b.shape==(200,len(e))
        assert np.allclose(archive[protocol+'__refit_effect'],b-a,atol=1e-13)
        for mode,z in [('frozen',a),('refit',b)]:
            w=old[(old.protocol==protocol)&(old['mode']==mode)].pivot(index='replicate',columns='session',values='coupling')
            assert np.allclose(z.sum(1),w.postop-w.preop,atol=1e-9)
            v=groups[(groups.protocol==protocol)&(groups['mode']==mode)&(groups.partition=='support')]
            assert np.isclose(v.mean_contribution.sum(),z.sum(1).mean(),atol=1e-13)
    check=pd.read_csv(OUT/'matched_reproduction.csv')
    assert len(check)==2400 and check.previous_delta_error.abs().max()<1e-9


def test_edge_investigation_preserves_all_primary_and_previous_matched_inputs():
    hashes=json.loads((OUT/'before_edge_analysis/unchanged_inputs.json').read_text())
    for name,expected in hashes.items():assert sha((RESULTS/'before_primary_sc'/name) if (RESULTS/'before_primary_sc'/name).exists() else ROOT/name)==expected,name
