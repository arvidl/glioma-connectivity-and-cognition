"""Portable CPU preprocessing: explicit rigid/affine transforms, no hidden downloads.

Transforms are SimpleITK fixed-to-moving pull maps in LPS millimetres.
DWI susceptibility/eddy correction is delegated to pat16.dwi (FSL).
BOLD static susceptibility correction is delegated to pat16.bold. Slice timing
remains uncorrected.
"""
import json, time
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage as ndi
from . import DATA, RESULTS, ROOT

sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(6)


def read(path):
    return sitk.ReadImage(str(path), sitk.sitkFloat32)


def from_array(array, ref):
    out = sitk.GetImageFromArray(np.asarray(array, np.float32).transpose(2,1,0).copy())
    out.CopyInformation(ref)
    return out


def array(image):
    return sitk.GetArrayFromImage(image).transpose(2,1,0)


def register(fixed, moving, affine=False, mask=None, iterations=100, initial=None, metric="mi"):
    if initial is None:
        initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    if affine:
        if isinstance(initial, sitk.CompositeTransform):
            initial = initial.GetBackTransform()
        t = sitk.AffineTransform(3)
        t.SetCenter(initial.GetCenter()); t.SetMatrix(initial.GetMatrix()); t.SetTranslation(initial.GetTranslation())
        initial = t
    r = sitk.ImageRegistrationMethod()
    if metric == "correlation":
        r.SetMetricAsCorrelation()
        r.SetMetricSamplingStrategy(r.NONE)
    else:
        r.SetMetricAsMattesMutualInformation(40)
        r.SetMetricSamplingStrategy(r.RANDOM)
        r.SetMetricSamplingPercentage(.15, seed=160)
    if mask is not None: r.SetMetricFixedMask(sitk.Cast(mask, sitk.sitkUInt8))
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=1., minStep=.002, numberOfIterations=iterations, relaxationFactor=.5, gradientMagnitudeTolerance=1e-7)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4,2,1])
    r.SetSmoothingSigmasPerLevel([2,1,0]); r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    r.SetInitialTransform(initial, inPlace=True)
    r.Execute(fixed, moving)
    return initial, {'metric':float(r.GetMetricValue()),'stop':r.GetOptimizerStopConditionDescription()}


def warp(moving, fixed, transform, labels=False):
    return sitk.Resample(moving, fixed, transform, sitk.sitkNearestNeighbor if labels else sitk.sitkLinear, 0., sitk.sitkFloat32)


def save(image, path):
    sitk.WriteImage(image, str(path))


def brain_mask(data):
    pos = data[data > 0]
    m = data > np.percentile(pos,35)
    m = ndi.binary_closing(m, iterations=2)
    labs,n=ndi.label(m)
    if n: m = labs == (np.bincount(labs.ravel())[1:].argmax()+1)
    return ndi.binary_fill_holes(m)


def anatomy():
    out=RESULTS/'anatomy';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists(): return
    pre=read(DATA/'sub-PAT16/ses-preop/anat/sub-PAT16_ses-preop_T1w.nii.gz')
    post=read(DATA/'sub-PAT16/ses-postop/anat/sub-PAT16_ses-postop_T1w.nii.gz')
    template=read(ROOT/'data/atlas/MNI152_T1_2mm.nii.gz')
    atlas=read(ROOT/'data/atlas/schaefer100.nii.gz')
    bm=read(ROOT/'data/atlas/MNI152_T1_2mm_brain_mask.nii.gz')
    # First rigid, then 12-parameter affine; template-to-patient sampling map.
    print('Registering MNI atlas to preop T1',flush=True)
    t0,q0=register(pre,template,iterations=100)
    tp,qp=register(pre,template,affine=True,initial=t0,iterations=100)
    sitk.WriteTransform(tp,str(out/'preop_to_mni_pull.tfm'))
    premask=warp(bm,pre,tp,True)
    tp,qref=register(pre*premask,read(ROOT/'data/atlas/MNI152_T1_2mm_brain.nii.gz'),affine=True,initial=tp,iterations=200)
    sitk.WriteTransform(tp,str(out/'preop_to_mni_pull.tfm'))
    premask=warp(bm,pre,tp,True)
    save(premask,out/'preop_brain_mask.nii.gz')
    masksource=read(DATA/'derivatives/tumor_masks/sub-PAT16/anat/sub-PAT16_space_T1_label-tumor_binary.nii.gz')
    # Header world coordinates, independently visual-QC'd; never copy voxel indices.
    lesion=warp(masksource,pre,sitk.Transform(3,sitk.sitkIdentity),True)
    save(lesion,out/'preop_lesion.nii.gz')
    costmask=from_array((array(premask)>.5)&~ndi.binary_dilation(array(lesion)>.5,iterations=5),pre)
    print('Registering postop T1 to preop T1 outside preop lesion',flush=True)
    tq,qq=register(pre,post,mask=costmask,iterations=120)
    sitk.WriteTransform(tq,str(out/'preop_to_postop_pull.tfm'))
    post_to_mni=sitk.CompositeTransform(3)
    post_to_mni.AddTransform(tp);post_to_mni.AddTransform(tq.GetInverse())
    for name,ref,t in [('preop',pre,tp),('postop',post,post_to_mni)]:
        save(warp(atlas,ref,t,True),out/f'{name}_atlas.nii.gz')
        save(warp(bm,ref,t,True),out/f'{name}_brain_mask.nii.gz')
    save(warp(post,pre,tq),out/'postop_in_preop.nii.gz')
    save(warp(lesion,post,tq.GetInverse(),True),out/'postop_preop_lesion_footprint.nii.gz')
    (out/'complete.json').write_text(json.dumps({'mni_rigid':q0,'mni_affine':qp,'brain_only_affine_refinement':qref,'longitudinal_rigid':qq,'postop_mask':'transported preop lesion footprint, NOT postoperative cavity'},indent=2))


def motion_correct(kind, session):
    """Dispatch DWI to FSL, or BOLD to rigid motion + static field correction."""
    if kind=='dwi':
        from .dwi import correct_diffusion
        return correct_diffusion(session)
    rigid_motion_correct(kind,session)
    from .bold import correct_bold
    return correct_bold(session)


def rigid_motion_correct(kind, session, out=None):
    """BOLD realignment; DWI option is only the matched uncorrected comparator."""
    out=RESULTS/session if out is None else Path(out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/f'{kind}_motion.json').exists(): return
    stem=f'sub-PAT16_ses-{session}'
    path=DATA/f'sub-PAT16/ses-{session}'/kind/(stem+('_acq-AP_dwi.nii.gz' if kind=='dwi' else '_task-rest_bold.nii.gz'))
    img=nib.load(path);data=img.get_fdata(dtype=np.float32)
    # Use a 3-D NIfTI round-trip to preserve full image geometry in SITK.
    if kind=='dwi':
        bval=np.loadtxt(str(path).replace('.nii.gz','.bval'))
        bvec=np.loadtxt(str(path).replace('.nii.gz','.bvec')).T
        refdata=data[...,bval<50].mean(-1)
    else: refdata=data[...,4:14].mean(-1)
    refpath=out/f'{kind}_reference.nii.gz';nib.save(nib.Nifti1Image(refdata,img.affine),refpath)
    ref=read(refpath)
    m=brain_mask(refdata);mask=from_array(m,ref)
    corrected=np.empty_like(data);params=[];metrics=[]
    D=np.array(ref.GetDirection()).reshape(3,3)
    corrected_bvec=[]
    transforms=[]
    start=time.time()
    for v in range(data.shape[-1]):
        moving=from_array(data[...,v],ref)
        t,q=register(ref,moving,mask=mask,iterations=80,metric="correlation" if kind=="func" else "mi")
        transforms.append(t)
        corrected[...,v]=array(warp(moving,ref,t))
        # Rigid Euler angles (rad) and translation (mm), in LPS.
        params.append(list(t.GetParameters()));metrics.append(q['metric'])
        if kind=='dwi':
            R=np.array(t.GetMatrix()).reshape(3,3)
            vec=D.T@R.T@D@bvec[v]
            corrected_bvec.append(vec/(np.linalg.norm(vec) or 1))
        if v%20==0: print(f'{session} {kind}: {v}/{data.shape[-1]}, {time.time()-start:.0f}s',flush=True)
    nib.save(nib.Nifti1Image(corrected,img.affine),out/f'{kind}_motion_corrected.nii.gz')
    np.savetxt(out/f'{kind}_motion.tsv',params,delimiter='\t',header='rx_rad\try_rad\trz_rad\ttx_mm\tty_mm\ttz_mm')
    for i,t in enumerate(transforms):
        (out/f'{kind}_transforms').mkdir(exist_ok=True)
        sitk.WriteTransform(t,str(out/f'{kind}_transforms/{i:03d}.tfm'))
    if kind=='dwi':
        np.savetxt(out/'dwi.bval',bval[None]);np.savetxt(out/'dwi.bvec',np.array(corrected_bvec).T)
    qc=coregister(kind,session,out)
    (out/f'{kind}_motion.json').write_text(json.dumps({'seconds':time.time()-start,'registration':qc,'frames':data.shape[-1], 'corrections':['rigid inter-volume realignment','DWI b-vector rotation' if kind=='dwi' else 'six motion parameters saved'],'not_performed':['susceptibility distortion correction','eddy current correction' if kind=='dwi' else 'slice timing correction'],'gradient_convention':'native LAS FSL voxel b-vectors; rotation in physical LPS then back to voxel axes'},indent=2))


def coregister(kind,session,out=None):
    out=RESULTS/session if out is None else Path(out)
    stem=f"sub-PAT16_ses-{session}"
    ref=read(out/f"{kind}_reference.nii.gz")
    # Scalar modality reference to T1 registration; transform is T1->EPI pull.
    t1=read(DATA/f'sub-PAT16/ses-{session}/anat/{stem}_T1w.nii.gz')
    bm=read(RESULTS/f'anatomy/{session}_brain_mask.nii.gz')
    from dipy.segment.mask import median_otsu
    masked,_=median_otsu(array(ref),median_radius=3,numpass=3,dilate=1)
    tc,qc=register(t1*bm,from_array(masked,ref),iterations=250)
    sitk.WriteTransform(tc,str(out/f't1_to_{kind}_pull.tfm'))
    atlas=read(RESULTS/f'anatomy/{session}_atlas.nii.gz')
    save(warp(atlas,ref,tc.GetInverse(),True),out/f'atlas_{kind}.nii.gz')
    save(warp(bm,ref,tc.GetInverse(),True),out/f'brain_mask_{kind}.nii.gz')
    save(warp(ref,t1,tc),out/f'{kind}_in_t1.nii.gz')
    return qc
