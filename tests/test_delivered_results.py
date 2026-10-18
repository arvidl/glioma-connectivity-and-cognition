"""Integration checks on the actual delivered patient-derived artifacts."""
import json
import numpy as np
import pandas as pd
from pat16 import DATA,RESULTS,SESSIONS


def test_paired_node_identity_counts_and_normalization():
    nodes=pd.read_csv(RESULTS/'nodes.csv')
    assert nodes.id.tolist()==list(range(1,101)) and nodes.label.nunique()==100
    for s in SESSIONS:
        c=np.load(RESULTS/s/'sc_counts.npy');w=np.load(RESULTS/s/'sc_fraction.npy')
        q=json.loads((RESULTS/s/'sc_qc.json').read_text())
        assert np.array_equal(c,c.T) and (np.diag(c)==0).all()
        assert np.triu(c,1).sum()==q['assigned_interparcel_streamlines']
        assert np.isclose(np.triu(w,1).sum(),1)
        assert (c>=0).all() and np.isfinite(w).all()


def test_fc_contract_and_saved_qc_match_data():
    common=np.load(RESULTS/'common_fc_nodes.npy')
    for s in SESSIONS:
        f=np.load(RESULTS/s/'fc_primary.npy');t=np.load(RESULTS/s/'roi_timeseries_clean.npy');keep=np.load(RESULTS/s/'fc_primary_keep.npy')
        q=json.loads((RESULTS/s/'fc_qc.json').read_text())['analyses'][0]
        assert f.shape==(100,100) and t.shape==(180,100)
        assert np.allclose(f,f.T,equal_nan=True) and np.nanmax(abs(f))<=1
        assert keep.sum()==q['retained_frames'] and q['residual_df']>=15
        empirical=np.corrcoef(t[keep][:,common],rowvar=False);np.fill_diagonal(empirical,0)
        assert np.allclose(f[np.ix_(common,common)],empirical,atol=1e-10)


def test_every_source_assessment_is_exported_without_synthetic_values():
    raw=pd.read_csv(DATA/'sub-PAT16/sub-PAT16_sessions.tsv',sep='\t',na_values=['n/a']).set_index('session_id')
    exported=pd.read_csv(RESULTS/'assessments_all.csv').set_index('measure')
    assert set(raw.columns)==set(exported.index)
    assert len(pd.read_csv(RESULTS/'cantab_all.csv'))==28
    for key in raw:
        for s in SESSIONS:
            a=raw.loc['ses-'+s,key];b=exported.loc[key,s]
            if pd.isna(a):assert pd.isna(b)
            else:
                try:assert np.isclose(float(a),float(b))
                except ValueError:assert a==b
