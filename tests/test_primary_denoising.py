"""Primary-model lineage, censor-specific fitting and archived-delivery integrity."""
import json
import numpy as np
import pandas as pd
import pytest
from pat16 import ROOT,RESULTS,SESSIONS
from pat16.bold import sha,variant_folder
from pat16.nuisance import OUT,subspace_similarity
from pat16.denoising import VERSION,primary_fit


def test_previous_delivery_and_spatial_structural_inputs_remain_intact():
    archive=RESULTS/'before_primary_anatomical5'
    manifest=json.loads((archive/'manifest.json').read_text())
    for r in manifest['files']:
        p=ROOT/r['path'] if r.get('immutable') else archive/r['path']
        assert sha(p)==r['sha256'],str(p)


@pytest.mark.parametrize('session',SESSIONS)
def test_primary_controls_use_censor_specific_fits_and_match_delivered_results(session):
    p=RESULTS/session;common=np.load(RESULTS/'common_fc_nodes.npy')
    sensitivity=pd.read_csv(RESULTS/'nuisance_anatomical_sensitivity.csv')
    for threshold,tag in [(.3,'fd03'),(.5,'primary'),(1.,'fd10')]:
        f,_,keep,q,_=primary_fit(session,threshold)
        assert np.allclose(f,np.load(p/f'fc_{tag}.npy'))
        assert np.array_equal(keep,np.load(p/f'fc_{tag}_keep.npy'))
        if threshold==.3:
            r=sensitivity[(sensitivity.session==session)&(sensitivity.BOLD_processing=='corrected')&
                (sensitivity.PVE_threshold==.95)&(sensitivity.components==5)&(sensitivity.FD_threshold==.3)].iloc[0]
            assert np.isclose(f[np.ix_(common,common)][np.triu_indices(common.sum(),1)].mean(),r.mean_fc_r)
            assert q['residual_df']==r.residual_df
    a=np.load(p/'primary_pcs_fd03.npy');b=np.load(p/'primary_nuisance_pcs.npy')
    assert subspace_similarity(a,b,np.load(p/'fc_fd03_keep.npy')).min()<.99
    gs=np.load(p/'primary_global_signal.npy')
    assert np.allclose(primary_fit(session,global_signal=gs)[0],np.load(p/'fc_gsr.npy'))


def test_bold_processing_comparisons_use_one_denoising_recipe():
    table=pd.read_csv(RESULTS/'bold_comparison.csv')
    assert set(table.denoising_model)=={'anatomical5'}
    common=np.load(RESULTS/'bold_comparison_common_nodes.npy');i,j=np.triu_indices(common.sum(),1)
    for _,r in table.iterrows():
        p=variant_folder(r.session,r.BOLD_processing) if r.BOLD_processing!='uncorrected' else OUT/r.session/'uncorrected'
        q=json.loads((p/'fc_qc.json').read_text())
        assert q['denoising_version']==VERSION and q['denoising_model']=='anatomical5'
        f=np.load(p/'fc_primary.npy')[np.ix_(common,common)]
        assert np.isclose(f[i,j].mean(),r.mean_fc_r)
        assert np.array_equal(np.load(p/'fc_primary_keep.npy'),np.load(RESULTS/r.session/'fc_primary_keep.npy'))
