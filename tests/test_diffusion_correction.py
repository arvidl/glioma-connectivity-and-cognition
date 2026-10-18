"""Validate metadata units, gradient lineage and actual FSL output contracts."""
import json,hashlib
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
from pat16 import ROOT,DATA,RESULTS,SESSIONS
from pat16.dwi import acquisition_row


def test_bids_phase_encoding_and_readout_are_not_filename_inferences():
    assert np.allclose(acquisition_row({'PhaseEncodingDirection':'j-','TotalReadoutTime':.0266003}),[0,-1,0,.0266003])
    assert np.allclose(acquisition_row({'PhaseEncodingDirection':'i','TotalReadoutTime':.02}),[1,0,0,.02])
    with pytest.raises(ValueError):acquisition_row({'PhaseEncodingDirection':'AP','TotalReadoutTime':.02})
    with pytest.raises(ValueError):acquisition_row({'PhaseEncodingDirection':'j','TotalReadoutTime':26.6})


@pytest.mark.parametrize('session',SESSIONS)
def test_corrected_data_and_gradients_keep_volume_lineage(session):
    out=RESULTS/session;folder=out/'correction'
    stem=DATA/f'sub-PAT16/ses-{session}/dwi/sub-PAT16_ses-{session}_acq-AP_dwi'
    raw=nib.load(str(stem)+'.nii.gz');corrected=nib.load(out/'dwi_corrected.nii.gz')
    assert corrected.shape==raw.shape==(96,96,60,102)
    assert np.allclose(corrected.affine,raw.affine,atol=1e-4)
    assert np.isfinite(corrected.get_fdata(dtype=np.float32)).all()
    b=np.loadtxt(out/'dwi.bval');v=np.loadtxt(out/'dwi.bvec')
    assert np.array_equal(b,np.loadtxt(str(stem)+'.bval'))
    assert v.shape==(3,102) and np.allclose(np.linalg.norm(v[:,b>50],axis=0),1,atol=1e-5)
    assert np.allclose(v,np.loadtxt(out/'dwi_corrected.eddy_rotated_bvecs'))
    assert np.allclose(v[:,b<50],0)
    command=json.loads((folder/'eddy_command.json').read_text())
    # Logs retain the original execution location after the package is cloned.
    # Check the complete project-relative identity and gradient sibling lineage,
    # then verify the relocated bytes against the unchanged input manifest.
    args=dict(arg.split('=',1) for arg in command[1:] if '=' in arg)
    original=Path(args['--imain']);relative=Path(str(stem.relative_to(ROOT))+'.nii.gz')
    assert original.parts[-len(relative.parts):]==relative.parts
    recorded_stem=str(original).removesuffix('.nii.gz')
    assert args['--bvecs']==recorded_stem+'.bvec' and args['--bvals']==recorded_stem+'.bval'
    manifest={r['path']:r['sha256'] for r in json.loads((ROOT/'data/manifest.json').read_text())['files']}
    for suffix in ['.nii.gz','.bvec','.bval']:
        current=Path(str(stem)+suffix)
        assert hashlib.sha256(current.read_bytes()).hexdigest()==manifest[str(current.relative_to(ROOT))]
    fitmask=nib.load(out/'structural_fit_mask.nii.gz').get_fdata()>.5
    ap_jac=nib.load(folder/'jacobian_01.nii.gz').get_fdata()
    pa_jac=nib.load(folder/'jacobian_07.nii.gz').get_fdata()
    assert fitmask.any() and not (fitmask&((ap_jac<=0)|(pa_jac<=0))).any()
    index=np.loadtxt(folder/'index.txt',dtype=int).ravel()
    assert len(index)==raw.shape[3] and (index==1).all()
    acqp=np.loadtxt(folder/'acqparams.txt')
    assert acqp.shape==(8,4) and (acqp[:6,1]==-1).all() and (acqp[6:,1]==1).all()
    assert np.allclose(acqp[:,3],.0266003)
    stack_image=nib.load(folder/'b0_AP_PA.nii.gz')
    stacked=stack_image.get_fdata(dtype=np.float32)
    # Original int16 header is retained; NIfTI scaling can requantize signal
    # by at most half a stored intensity step, without spatial resampling.
    tolerance=abs(stack_image.dataobj.slope)/2+np.finfo(np.float32).eps*np.max(abs(raw.dataobj[...,0]))
    assert np.allclose(stacked[...,0],raw.dataobj[...,0],rtol=0,atol=tolerance)
    assert nib.load(folder/'b0_unwarped.nii.gz').shape==stacked.shape
    q=json.loads((out/'sc_qc.json').read_text())
    assert q['preprocessing']=='corrected'
    assert q['tensor_bvalues_s_mm2']==[0,700,1200]
    assert q['csd_bvalues_s_mm2']==[0,2800] and q['csd_volumes']==56
    # The historical diffusion-only stage left BOLD unchanged; preserve that
    # baseline when subsequently correcting BOLD with its own readout/geometry.
    assert np.array_equal(np.load(RESULTS/'bold_uncorrected'/session/'fc_primary.npy'),np.load(RESULTS/'legacy_baseline'/session/'fc_primary.npy'),equal_nan=True)
