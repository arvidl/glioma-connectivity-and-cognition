"""Scientific safeguards for the new anatomical nuisance-model comparison."""
import json
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from pat16 import ROOT,DATA,RESULTS,SESSIONS
from pat16.bold import sha
from pat16.nuisance import OUT,pca_signals,subspace_similarity
from pat16.connectivity import clean_timeseries


def test_retained_frame_pca_does_not_train_on_censored_values():
    rng=np.random.default_rng(28)
    data=rng.normal(size=(3,4,5,180));mask=np.ones((3,4,5),bool)
    keep=np.ones(180,bool);keep[:4]=False;keep[60:66]=False
    altered=data.copy();altered[...,~keep]+=rng.normal(size=altered[...,~keep].shape)*1e4
    a,va=pca_signals(data,mask,keep,True,5)
    b,vb=pca_signals(altered,mask,keep,True,5)
    assert subspace_similarity(a,b,keep).min()>1-1e-10
    assert np.allclose(va,vb)


def test_nuisance_basis_rotation_and_sign_do_not_change_fc():
    rng=np.random.default_rng(161);y=rng.normal(size=(180,9));pcs=rng.normal(size=(180,5))
    motion=rng.normal(size=(180,6))*1e-5
    rotation=np.linalg.qr(rng.normal(size=(5,5)))[0]
    fa,ra,ka,qa,_=clean_timeseries(y,motion,pcs)
    fb,rb,kb,qb,_=clean_timeseries(y,motion,pcs@rotation)
    assert np.allclose(fa,fb) and np.allclose(ra[ka],rb[kb])
    assert np.array_equal(ka,kb) and qa['residual_df']==qb['residual_df']


@pytest.mark.parametrize('session',SESSIONS)
def test_saved_anatomical_masks_models_and_primary_lineage(session):
    p=OUT/session/'corrected'
    wm=nib.load(p/'wm_mask_p95.nii.gz');csf=nib.load(p/'csf_mask_p95.nii.gz')
    valid=nib.load(RESULTS/session/'bold_valid_mask.nii.gz')
    assert wm.shape==csf.shape==valid.shape
    assert np.allclose(wm.affine,valid.affine) and np.allclose(csf.affine,valid.affine)
    w=wm.get_fdata()>.5;c=csf.get_fdata()>.5;v=valid.get_fdata()>.5
    assert not (w&c).any() and not ((w|c)&~v).any()
    for name,m in [('wm',w),('csf',c)]:
        assert m.sum()>20
        assert (nib.load(p/f'{name}_pve_BOLD.nii.gz').get_fdata()[m]>=.95-1e-7).all()
    old_prov=json.loads((OUT/'before_anatomical_comparison/analysis_provenance.json').read_text())
    original={r['path']:r['sha256'] for r in old_prov['derived_outputs']}
    for file in ['sc_fraction.npy','fc_primary_keep.npy','func_motion.tsv']:
        assert sha(RESULTS/session/file)==original[f'results/{session}/{file}']
    y=np.load(RESULTS/session/'roi_timeseries_raw.npy');motion=np.loadtxt(RESULTS/session/'func_motion.tsv')
    common=np.load(RESULTS/'common_fc_nodes.npy')
    for model in ['original5','anatomical5_raw','anatomical5','wm5_csf5','wm_csf_means']:
        f,_,keep,q,_=clean_timeseries(y,motion,np.load(p/f'{model}_regressors.npy'))
        assert np.array_equal(keep,np.load(RESULTS/session/'fc_primary_keep.npy'))
        assert np.allclose(f[np.ix_(common,common)],np.load(p/f'fc_{model}.npy'))
        assert q['residual_df']>=15
    assert np.allclose(np.load(p/'fc_anatomical5.npy'),np.load(RESULTS/session/'fc_primary.npy'))
    archive=RESULTS/'before_primary_anatomical5/results'/session
    assert np.allclose(np.load(p/'fc_original5.npy'),np.load(archive/'fc_primary.npy'))
    assert sha(archive/'fc_primary.npy')==original[f'results/{session}/fc_primary.npy']


def test_assessment_dictionary_coverage_and_ambiguous_fields():
    annotations=json.loads((ROOT/'config/assessment_annotations.json').read_text())
    raw=pd.read_csv(DATA/'sub-PAT16/sub-PAT16_sessions.tsv',sep='\t').set_index('session_id')
    exported=pd.read_csv(RESULTS/'assessments_all.csv').set_index('measure')
    assert set(annotations)==set(raw.columns)==set(exported.index) and len(annotations)==47
    assert exported[['definition','units','direction_text','caveat']].notna().all().all()
    sd=exported.index.str.endswith('_sd');assert exported.loc[sd,'favorable_direction'].isna().all()
    assert 'count' in annotations['soc_prob_min_moves']['caveat']
    assert 'mean or median' in annotations['rti_five_rt']['caveat']
    assert 'mapping is absent' in annotations['education_level']['caveat']
    assert 'not the two standard' in annotations['erq_processing']['caveat']
    assert pd.isna(exported.loc['sport_vs_preop','preop'])
