"""Auditable anatomical nuisance masks, retained-frame PCA and diagnostics."""
import json
import os
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.linalg import eigh, lstsq, orth, subspace_angles
from . import ROOT, DATA, RESULTS, SESSIONS
from .bold import variant_folder, sha
from .imaging import read, array, warp
from .connectivity import clean_timeseries

OUT = RESULTS/'nuisance_investigation'
METHOD = 'anatomical-nuisance-v1'


def save_array(x, ref, path):
    nib.save(nib.Nifti1Image(np.asarray(x), ref.affine), path)


def segment(session, force=False):
    """FAST T1 PVE, with a 5-mm lesion-footprint exclusion; native geometry."""
    out = OUT/session; out.mkdir(parents=True, exist_ok=True)
    qpath = out/'segmentation.json'
    if qpath.exists() and not force:
        return
    t1path = DATA/f'sub-PAT16/ses-{session}/anat/sub-PAT16_ses-{session}_T1w.nii.gz'
    img = nib.load(t1path)
    brain = nib.load(RESULTS/f'anatomy/{session}_brain_mask.nii.gz').get_fdata() > .5
    lesion_path = RESULTS/'anatomy'/('preop_lesion.nii.gz' if session=='preop' else 'postop_preop_lesion_footprint.nii.gz')
    lesion = nib.load(lesion_path).get_fdata() > .5
    mask = brain & (img.get_fdata(dtype=np.float32)>0) & (ndi.distance_transform_edt(~lesion, sampling=img.header.get_zooms()[:3]) > 5.)
    input_path = out/'T1_brain_lesion_excluded.nii.gz'
    save_array(img.get_fdata(dtype=np.float32)*mask, img, input_path)
    save_array(mask.astype('uint8'), img, out/'segmentation_input_mask.nii.gz')
    fsldir = Path(os.environ.get('FSLDIR', str(Path.home()/'fsl')))
    binary = fsldir/'share/fsl/bin/fast'
    if not binary.exists():
        import shutil
        found = shutil.which('fast')
        if found is None: raise RuntimeError('FAST is required only for raw tissue segmentation; set FSLDIR')
        binary = Path(found)
    command = [str(binary), '-t','1','-n','3','-H','0.1','-I','4','-l','20','-b','-B','-p', '-o',str(out/'tissue'),str(input_path)]
    # A completed external invocation with the identical fixed configuration may
    # be adopted, but only after checking its output geometry and probability sum.
    started = time.time()
    if force or not (out/'tissue_pve_2.nii.gz').exists():
        env = os.environ.copy(); env.update(FSLDIR=str(fsldir), FSLOUTPUTTYPE='NIFTI_GZ', OMP_NUM_THREADS='6')
        with (out/'fast.log').open('w') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    pves = [nib.load(out/f'tissue_pve_{i}.nii.gz') for i in range(3)]
    for p in pves:
        if p.shape != img.shape or not np.allclose(p.affine,img.affine): raise RuntimeError('FAST changed native geometry')
    a = np.stack([p.get_fdata(dtype=np.float32) for p in pves])
    if not np.isfinite(a).all() or np.max(abs(a.sum(0)[mask]-1)) > .002: raise RuntimeError('Invalid tissue fractions')
    q = {'method_version':METHOD,'software':'FSL FAST, 6.0.7.19 installation; binary ID 2412.6-dirty',
         'command':command,'binary_sha256':sha(binary),'input_sha256':sha(t1path),'input_mask_sha256':sha(out/'segmentation_input_mask.nii.gz'),
         'classes':{'pve_0':'CSF','pve_1':'GM','pve_2':'WM'},'lesion_exclusion_mm':5.,
         'brain_mask':'Existing template-derived session T1 brain mask; not an expert segmentation',
         'bias_correction':True,'seconds_in_wrapper':time.time()-started,
         'limitations':['T1-only unsupervised segmentation, not manually validated','Post-op exclusion is transported pre-op footprint, not a cavity segmentation']}
    qpath.write_text(json.dumps(q,indent=2))


def legacy_mask(session, variant):
    p = variant_folder(session,variant)
    brain = nib.load(p/'brain_mask_func.nii.gz').get_fdata() > .5
    if variant=='corrected': brain &= nib.load(p/'bold_valid_mask.nii.gz').get_fdata() > .5
    atlas = nib.load(p/'atlas_func.nii.gz').get_fdata() > 0
    lesion = nib.load(p/'preop_lesion_footprint_func.nii.gz').get_fdata() > .5
    return ndi.binary_erosion(brain,iterations=2) & ~ndi.binary_dilation(atlas,iterations=1) & ~ndi.binary_dilation(lesion,iterations=1)


def anatomical_masks(session, variant='corrected', threshold=.95):
    """High-PVE, 1-mm-eroded T1 masks resampled to each BOLD reference.

    Both nearest-neighbour native-core membership and linearly interpolated PVE
    must pass. Require 2 mm inside the T1 brain and >5 mm from the footprint.
    """
    p = variant_folder(session,variant); out = OUT/session/variant
    out.mkdir(parents=True,exist_ok=True)
    ref = read(p/'func_reference.nii.gz'); ni = nib.load(p/'func_reference.nii.gz')
    pull = sitk.ReadTransform(str(p/'t1_to_func_pull.tfm')).GetInverse()
    bm = read(RESULTS/f'anatomy/{session}_brain_mask.nii.gz')
    brain = array(bm) > .5
    inside = ndi.distance_transform_edt(brain,sampling=bm.GetSpacing()) > 2.
    support = nib.load(p/'brain_mask_func.nii.gz').get_fdata() > .5
    if variant!='uncorrected': support &= nib.load(p/'bold_valid_mask.nii.gz').get_fdata() > .5
    lesion = nib.load(p/'preop_lesion_footprint_func.nii.gz').get_fdata() > .5
    support &= ~ndi.binary_dilation(lesion,iterations=1)
    masks = {}; fractions = {}
    for tissue,i in [('CSF',0),('GM',1),('WM',2)]:
        src = read(OUT/session/f'tissue_pve_{i}.nii.gz')
        values = array(src)
        fractions[tissue] = array(warp(src,ref,pull))
        if tissue=='GM': continue
        native = (values>=threshold) & inside
        # Physical erosion at native 1-mm resolution, before EPI resampling.
        native &= ndi.distance_transform_edt(native,sampling=src.GetSpacing()) > 1.
        core = sitk.GetImageFromArray(native.transpose(2,1,0).astype('uint8'));core.CopyInformation(src)
        mask = (array(warp(core,ref,pull,True))>.5) & (fractions[tissue]>=threshold) & support
        if mask.sum()<20: raise RuntimeError(f'{session}/{variant}/{tissue}: insufficient pure tissue voxels at {threshold}')
        masks[tissue] = mask
        save_array(mask.astype('uint8'),ni,out/f'{tissue.lower()}_mask_p{round(threshold*100):02d}.nii.gz')
        save_array(native.astype('uint8'),nib.load(OUT/session/f'tissue_pve_{i}.nii.gz'),OUT/session/f'{tissue.lower()}_T1_core_p{round(threshold*100):02d}.nii.gz')
    for tissue,f in fractions.items():save_array(f.astype('float32'),ni,out/f'{tissue.lower()}_pve_BOLD.nii.gz')
    return masks, fractions


def temporal_basis(nt, tr=2.4):
    t=np.arange(nt,dtype=float)
    k=np.arange(1,nt);k=k[k/(2*nt*tr)<.01]
    return np.column_stack([np.ones(nt),t,np.cos(np.pi*(t[:,None]+.5)*k/nt)])


def pca_signals(data,mask,keep=None,temporal=False,n=20):
    """Return full-time scores fitted only on keep, plus variance fractions.

    temporal=True removes constant, linear and <0.01-Hz DCT terms first,
    fitting their coefficients and voxel scaling on retained original indices.
    raw/all-frame PCA exactly reproduces the historical mask's algorithm.
    """
    z=data[mask].T.astype(float)
    if keep is None:keep=np.ones(z.shape[0],bool)
    if temporal:
        X=temporal_basis(len(z));z-=X@lstsq(X[keep],z[keep],cond=1e-8)[0]
    z-=z[keep].mean(0)
    sd=z[keep].std(0);z=z[:,sd>1e-5]/sd[sd>1e-5]
    if z.shape[1]<n:raise RuntimeError('Insufficient variable voxels for PCA')
    eigen,u=eigh(z[keep]@z[keep].T)
    order=np.argsort(eigen)[::-1];eigen=np.maximum(eigen[order],0);u=u[:,order]
    n=min(n,int((eigen>eigen[0]*1e-12).sum()))
    weights=z[keep].T@(u[:,:n]/np.sqrt(eigen[:n]))
    scores=z@weights
    return scores,eigen[:n]/eigen.sum()


def subspace_similarity(a,b,keep):
    x=orth(a[keep]);y=orth(b[keep])
    return np.cos(subspace_angles(x,y))**2


def base_residuals(signals,motion):
    return clean_timeseries(signals,motion,np.empty((len(signals),0)))[1]


def linear_sampling_audit(session):
    """Check the matched sampler against saved linear data on five raw frames."""
    from .bold import sampling_map
    raw=DATA/f'sub-PAT16/ses-{session}/func/sub-PAT16_ses-{session}_task-rest_bold.nii.gz'
    img=nib.load(raw);data=img.get_fdata(dtype=np.float32)
    old=nib.load(variant_folder(session,'uncorrected')/'func_motion_corrected.nii.gz').get_fdata(dtype=np.float32)
    mask=legacy_mask(session,'uncorrected');errors=[]
    for v in [4,40,90,130,179]:
        motion=sitk.ReadTransform(str(RESULTS/session/f'func_transforms/{v:03d}.tfm'))
        coords,_=sampling_map(data.shape[:3],img.affine,motion,np.zeros(data.shape[:3]),1)
        values=ndi.map_coordinates(data[...,v],coords,order=1,mode='constant',cval=0.)
        region=mask.copy()
        for i,size in enumerate(data.shape[:3]):region &= (coords[i]>=1)&(coords[i]<=size-2)
        errors.append(float(np.sqrt(np.mean((values[region]-old[...,v][region])**2))/np.sqrt(np.mean(old[...,v][region]**2))))
    return {'session':session,'tested_frames':'4,40,90,130,179','max_linear_resampling_relative_RMSE':max(errors),
            'interpretation':'Composed order-1 sampling versus historical SimpleITK, interior nuisance voxels'}


def cubic_motion_control(session):
    """Matched cubic interpolation from raw BOLD with susceptibility shift zero."""
    from .bold import unwarp_volume
    path=OUT/session/'motion_cubic_zero_field.nii.gz'
    raw=DATA/f'sub-PAT16/ses-{session}/func/sub-PAT16_ses-{session}_task-rest_bold.nii.gz'
    import hashlib
    transform_id=hashlib.sha256(''.join(sha(RESULTS/session/f'func_transforms/{v:03d}.tfm') for v in range(180)).encode()).hexdigest()
    lineage={'raw_BOLD_sha256':sha(raw),'motion_transforms_sha256':transform_id,
             'motion_parameters_sha256':sha(RESULTS/session/'func_motion.tsv')}
    qpath=path.with_suffix('').with_suffix('.json')
    if path.exists() and qpath.exists() and all(json.loads(qpath.read_text()).get(k)==v for k,v in lineage.items()):
        return nib.load(path).get_fdata(dtype=np.float32)
    img=nib.load(raw);data=img.get_fdata(dtype=np.float32);out=np.empty_like(data)
    zero=np.zeros(data.shape[:3],dtype=np.float32)
    for v in range(data.shape[-1]):
        transform=sitk.ReadTransform(str(RESULTS/session/f'func_transforms/{v:03d}.tfm'))
        out[...,v],_,_=unwarp_volume(data[...,v],img.affine,transform,zero,1)
    save_array(out,img,path)
    qpath.write_text(json.dumps({'purpose':'Matched interpolation diagnostic only; not a primary BOLD pipeline',**lineage,
        'sampling':'same composed sampler, cubic, zero susceptibility shift, identical saved motion'},indent=2))
    return out
