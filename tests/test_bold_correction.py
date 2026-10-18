"""Known-deformation phantom, coordinate conventions, and delivered BOLD lineage."""
import json
import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage as ndi
from pat16 import RESULTS, SESSIONS
from pat16.bold import signed_shift, unwarp_volume, sampling_map, sha, variant_folder


@pytest.mark.parametrize('direction', ['j', 'j-'])
def test_known_stretch_and_translation_phantom_recovers_intensity(direction):
    shape = (24, 48, 22)
    x, y, z = np.indices(shape, dtype=float)
    phantom = np.exp(-((x-11)**2/24+(y-23)**2/70+(z-10)**2/18))
    sign = -1 if direction.endswith('-') else 1
    slope, offset = sign*.09, sign*1.6
    shift = slope*(y-23)+offset
    # Analytic forward image: yd = (1+slope)*yu + offset - slope*23.
    yu = (y-offset+slope*23)/(1+slope)
    distorted = np.exp(-((x-11)**2/24+(yu-23)**2/70+(z-10)**2/18))/(1+slope)
    field = shift/(sign*.02)
    actual_shift, axis = signed_shift(field, {'PhaseEncodingDirection':direction, 'TotalReadoutTime':.02})
    corrected, valid, jac = unwarp_volume(distorted, np.eye(4), sitk.Transform(3, sitk.sitkIdentity), actual_shift, axis)
    region = valid&(x>3)&(x<20)&(y>8)&(y<39)&(z>3)&(z<18)
    error = np.sqrt(np.mean((corrected[region]-phantom[region])**2))
    uncorrected_error = np.sqrt(np.mean((distorted[region]-phantom[region])**2))
    wrong, _, _ = unwarp_volume(distorted, np.eye(4), sitk.Transform(3, sitk.sitkIdentity), -actual_shift, axis)
    assert error < .0002 and error < uncorrected_error/50
    assert np.sqrt(np.mean((wrong[region]-phantom[region])**2)) > 50*error
    assert np.allclose(jac, 1+slope)


def test_oblique_las_motion_and_pe_composition_has_physical_direction_and_jacobian():
    shape = (12, 13, 14)
    affine = np.array([[-3,0,0,17],[0,2.98,-.35,-22],[0,.35,2.98,-11],[0,0,0,1.]])
    motion = sitk.Euler3DTransform();motion.SetRotation(.08,-.05,.12);motion.SetTranslation((1.2,-.8,.4))
    x,y,z = np.indices(shape, dtype=float);shift = .07*y+.03*x-.02*z+.2
    coordinates, jac = sampling_map(shape, affine, motion, shift, 1)
    voxel = np.array([5,6,7]);lps = np.diag([-1.,-1.,1.,1.])@affine
    physical = (lps@np.r_[voxel,1])[:3]
    expected = np.linalg.inv(lps)@np.r_[motion.TransformPoint(tuple(physical)),1]
    expected[1] += shift[tuple(voxel)]
    assert np.allclose(coordinates[:,5,6,7], expected[:3])
    # Numerical differentiation of the full 3D pull, independent of the
    # matrix-determinant-lemma implementation used for production Jacobians.
    numerical = np.stack([np.stack(np.gradient(coordinates[i]),axis=-1) for i in range(3)],axis=-2)
    assert np.allclose(jac[2:-2,2:-2,2:-2],np.linalg.det(numerical)[2:-2,2:-2,2:-2],atol=1e-10)


@pytest.mark.parametrize('session', SESSIONS)
def test_bold_sdc_preserves_time_motion_and_sc_but_recomputes_fc(session):
    out = RESULTS/session;before = variant_folder(session,'uncorrected')
    q = json.loads((out/'bold_sdc.json').read_text())
    image = nib.load(out/'func_sdc.nii.gz');raw = nib.load(before/'func_motion_corrected.nii.gz')
    assert image.shape == raw.shape == (64,64,42,180)
    assert np.allclose(image.affine,raw.affine)
    assert np.isfinite(image.get_fdata(dtype=np.float32)).all()
    assert sha(out/'func_motion.tsv') == sha(before/'func_motion.tsv')
    assert sha(out/'sc_counts.npy') == sha(before/'sc_counts.npy') == q['SC_counts_sha256']
    assert np.array_equal(np.load(out/'fc_primary_keep.npy'),np.load(before/'fc_primary_keep.npy'))
    assert not np.allclose(np.load(out/'fc_primary.npy'),np.load(before/'fc_primary.npy'),equal_nan=True)
    for variant in ('corrected','readout_alternative'):
        p = variant_folder(session,variant)
        mask = nib.load(p/'bold_valid_mask.nii.gz').get_fdata()>.5
        jac = nib.load(p/'bold_min_jacobian.nii.gz').get_fdata()
        assert mask.any() and (jac[mask]>0).all()
        qc = json.loads((p/'fc_qc.json').read_text())
        assert qc['preprocessing']==variant and qc['image_sha256']==sha(p/'func_sdc.nii.gz')
    assert 0 < q['timing_alternative']['displacement_scale_relative_to_primary'] < 1
