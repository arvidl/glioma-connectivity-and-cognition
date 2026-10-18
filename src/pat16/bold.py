"""Static AP/PA field transfer and one-interpolation BOLD unwarping.

Field values are scalars in Hz, resampled through a rigid BOLD-to-b0 pull map.
Raw-volume sampling composes saved motion with a signed PE displacement. The
rank-one determinant accounts for the motion rotation as well as field gradient.
This is a small transparent implementation, not a run of fMRIPrep/SDCFlows.
"""
import hashlib
import json
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from . import DATA, RESULTS, ROOT
from .dwi import acquisition_row
from .imaging import read, array, from_array, register, warp, coregister

METHOD = 'static-topup-transfer-v1'
VARIANTS = ('corrected', 'uncorrected', 'readout_alternative')


def variant_folder(session, variant='corrected'):
    if variant not in VARIANTS:
        raise ValueError(variant)
    return RESULTS/session if variant == 'corrected' else RESULTS/('bold_'+variant)/session


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preserve_uncorrected(session):
    """Freeze the delivered rigid-only BOLD stage before replacing its reference."""
    source = RESULTS/session
    out = variant_folder(session, 'uncorrected')
    if (out/'preserved.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    qcpath = source/'fc_qc.json'
    if qcpath.exists() and json.loads(qcpath.read_text()).get('preprocessing') == 'corrected':
        raise RuntimeError('Cannot archive corrected FC as the uncorrected baseline')
    files = set()
    for pattern in ['func_*', 'atlas_func.nii.gz', 'brain_mask_func.nii.gz', 't1_to_func_pull.tfm',
                    'preop_lesion_footprint_func.nii.gz', 'fc_*', 'fd.npy', 'roi_timeseries*',
                    'nuisance_pcs.npy', 'sc_counts.npy', 'sc_fraction.npy', 'sc_qc.json']:
        files.update(source.glob(pattern))
    records = []
    for p in sorted(files):
        if p.is_dir():
            shutil.copytree(p, out/p.name, dirs_exist_ok=True)
        else:
            shutil.copy2(p, out/p.name)
            records.append({'path': p.name, 'sha256': sha(p)})
    (out/'preserved.json').write_text(json.dumps(records, indent=2))
    archive = RESULTS/'bold_uncorrected'
    if qcpath.exists() and not (archive/'report_before_BOLD_SDC.pdf').exists():
        if (ROOT/'report/main.pdf').exists():
            shutil.copy2(ROOT/'report/main.pdf', archive/'report_before_BOLD_SDC.pdf')
        for name in ['summary.json', 'graph_metrics.csv', 'network_summary.csv', 'sensitivity.csv']:
            if (RESULTS/name).exists():
                shutil.copy2(RESULTS/name, archive/name)


def signed_shift(field_hz, metadata, scale=1.):
    row = acquisition_row(metadata)
    axis = int(np.flatnonzero(row[:3])[0])
    return np.asarray(field_hz)*row[axis]*row[3]*scale, axis


def sampling_map(shape, affine, motion, shift, axis):
    """Output undistorted voxels -> raw moving voxels; return exact determinant."""
    origin = np.array(motion.TransformPoint((0., 0., 0.)))
    linear = np.column_stack([np.array(motion.TransformPoint(tuple(v)))-origin for v in np.eye(3)])
    physical = np.eye(4); physical[:3, :3] = linear; physical[:3, 3] = origin
    lps_affine = np.diag([-1., -1., 1., 1.])@affine
    m = np.linalg.inv(lps_affine)@physical@lps_affine
    grid = np.indices(shape, dtype=np.float64).reshape(3, -1)
    coordinates = m[:3, :3]@grid+m[:3, 3, None]
    coordinates[axis] += shift.ravel()
    gradient = np.stack(np.gradient(shift), axis=-1)
    jac = np.linalg.det(m[:3, :3])*(1+gradient@np.linalg.inv(m[:3, :3])[:, axis])
    return coordinates.reshape((3,)+tuple(shape)), jac


def unwarp_volume(data, affine, motion, shift, axis):
    coords, jac = sampling_map(data.shape, affine, motion, shift, axis)
    valid = np.isfinite(jac)&(jac > 0)
    for i, n in enumerate(data.shape):
        valid &= (coords[i] >= 1)&(coords[i] <= n-2)
    output = ndi.map_coordinates(data, coords, order=3, mode='constant', cval=0., output=np.float32)
    output *= np.where(jac > 0, jac, 0).astype(np.float32)
    return output, valid, jac


def correct_bold(session):
    out = RESULTS/session
    qcpath = out/'bold_sdc.json'
    if qcpath.exists() and json.loads(qcpath.read_text()).get('method_version') == METHOD:
        return
    start = time.time()
    preserve_uncorrected(session)
    baseline = variant_folder(session, 'uncorrected')
    folder = out/'bold_sdc'; folder.mkdir(exist_ok=True)
    rawpath = DATA/f'sub-PAT16/ses-{session}/func/sub-PAT16_ses-{session}_task-rest_bold.nii.gz'
    rawimg = nib.load(rawpath); data = rawimg.get_fdata(dtype=np.float32)
    meta = json.loads(Path(str(rawpath).replace('.nii.gz', '.json')).read_text())
    dm = json.loads((DATA/f'sub-PAT16/ses-{session}/dwi/sub-PAT16_ses-{session}_acq-AP_dwi.json').read_text())
    # Rigid motion was estimated against this exact distorted reference.
    rawref = from_array(data[..., 4:14].mean(3), read(baseline/'func_reference.nii.gz'))
    b0 = read(out/'dwi_reference.nii.gz')
    field = read(out/'correction/field_Hz.nii.gz')
    bmask = read(out/'correction/brain_mask.nii.gz')
    fmask = array(read(baseline/'brain_mask_func.nii.gz')) > .5
    lesion_name = 'preop_lesion' if session == 'preop' else 'postop_preop_lesion_footprint'
    tc = sitk.ReadTransform(str(baseline/'t1_to_func_pull.tfm'))
    lesion = array(warp(read(RESULTS/f'anatomy/{lesion_name}.nii.gz'), rawref, tc.GetInverse(), True)) > .5
    fitmask = from_array(fmask&~ndi.binary_dilation(lesion, iterations=1), rawref)
    moving = b0*bmask
    transform, quality = register(rawref*from_array(fmask, rawref), moving, mask=fitmask, iterations=200)
    history = [quality]
    identity = sitk.Transform(3, sitk.sitkIdentity)
    for iteration in range(2):
        f = array(warp(field, rawref, transform))
        shift, axis = signed_shift(f, meta)
        corrected_ref, _, _ = unwarp_volume(array(rawref), rawimg.affine, identity, shift, axis)
        transform, quality = register(from_array(corrected_ref*fmask, rawref), moving,
                                      mask=fitmask, iterations=200, initial=transform)
        history.append(quality)
    sitk.WriteTransform(transform, str(folder/'bold_to_b0_pull.tfm'))
    transferred = warp(field, rawref, transform)
    sitk.WriteImage(transferred, str(folder/'field_BOLD_Hz.nii.gz'))
    source_support = array(warp(bmask, rawref, transform, True)) > .5
    # Alternative timing interpretation: change the displacement ratio, not the
    # already fitted DWI field. The primary uses supplied TRT in both scans.
    d_alt = dm['EffectiveEchoSpacing']*(dm['PhaseEncodingLines']-1)
    b_alt = meta['EffectiveEchoSpacing']*(meta['PhaseEncodingLines']-1)
    alpha = (b_alt/meta['TotalReadoutTime'])*(dm['TotalReadoutTime']/d_alt)
    histories = {}
    for variant, scale in [('corrected', 1.), ('readout_alternative', alpha)]:
        dest = variant_folder(session, variant); dest.mkdir(parents=True, exist_ok=True)
        shift, axis = signed_shift(array(transferred), meta, scale)
        fixedvalid = source_support.copy()
        corrected = np.empty_like(data)
        min_jac = np.full(data.shape[:3], np.inf, dtype=np.float32)
        for v in range(data.shape[3]):
            motion = sitk.ReadTransform(str(out/f'func_transforms/{v:03d}.tfm'))
            corrected[..., v], good, jac = unwarp_volume(data[..., v], rawimg.affine, motion, shift, axis)
            fixedvalid &= good; min_jac = np.minimum(min_jac, jac)
        for name, x in [('func_sdc', corrected), ('func_reference', corrected[..., 4:14].mean(3)),
                        ('bold_valid_mask', fixedvalid.astype('uint8')), ('bold_min_jacobian', min_jac),
                        ('bold_shift_vox', shift)]:
            header = rawimg.header.copy(); header.set_data_dtype(x.dtype)
            nib.save(nib.Nifti1Image(x, rawimg.affine, header), dest/f'{name}.nii.gz')
        if dest != out:
            shutil.copy2(out/'func_motion.tsv', dest/'func_motion.tsv')
        registration = coregister('func', session, dest)
        histories[variant] = {'field_scale': float(scale), 'registration': registration,
            'valid_voxels': int(fixedvalid.sum()),
            'reference_brain_covered_fraction': float((fixedvalid&fmask).sum()/fmask.sum()),
            'nonpositive_min_jacobian_in_reference_brain': int(((min_jac <= 0)&fmask).sum()),
            'signed_shift_vox_brain_percentiles_1_50_99': np.percentile(shift[fmask&source_support], [1, 50, 99]).tolist()}
        # Invalidate derived FC after changing its image/atlas; no stale variant.
        (dest/'fc_qc.json').unlink(missing_ok=True)
    qc = {'method_version': METHOD, 'method': 'Rigidly transferred topup field; static PE pull + motion; cubic single interpolation',
        'frames': data.shape[3], 'phase_encoding': meta['PhaseEncodingDirection'],
        'BOLD_total_readout_seconds': meta['TotalReadoutTime'], 'DWI_total_readout_seconds': dm['TotalReadoutTime'],
        'timing_alternative': {'DWI_echo_spacing_times_lines_minus_one': d_alt, 'BOLD_echo_spacing_times_lines_minus_one': b_alt,
                               'displacement_scale_relative_to_primary': float(alpha), 'registration': 'Held fixed to isolate timing amplitude'},
        'field_registration_iterations': history, 'variants': histories,
        'source_field_sha256': sha(out/'correction/field_Hz.nii.gz'), 'raw_BOLD_sha256': sha(rawpath),
        'motion_parameters_sha256': sha(out/'func_motion.tsv'), 'SC_counts_sha256': sha(out/'sc_counts.npy'),
        'coordinates': 'SITK LPS physical pulls; BIDS signed voxel j; motion first, then raw PE-axis displacement',
        'jacobian': 'Exact rank-one determinant of composed affine-motion + PE pull; nonpositive values excluded',
        'readout_decision': 'Primary uses supplied TRT ratio; alternative uses echo-spacing/phase-lines ratio. Hz scale is conditional.',
        'limitations': ['No BOLD-specific reverse-PE scan or shim settings; cross-acquisition static field assumption',
                        'BOLD precedes DWI by about 9-10 minutes; possible shim/field drift',
                        'No susceptibility-by-motion, slice-to-volume or slice-timing correction',
                        'GE-EPI signal dropout and neurovascular uncoupling cannot be recovered',
                        'Rigid field transfer and timing sensitivity are research approximations, not independent validation'],
        'seconds': time.time()-start}
    qcpath.write_text(json.dumps(qc, indent=2))
    print(session, 'BOLD susceptibility correction complete', histories, flush=True)
