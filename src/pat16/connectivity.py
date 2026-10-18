"""Patient-derived SC/FC with fixed node identity and explicit missingness."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.linalg import lstsq
from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.data import default_sphere
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.utils import random_seeds_from_mask
from dipy.tracking.streamline import length
from . import RESULTS, ROOT
from .imaging import read, warp, save, array
import SimpleITK as sitk

N=100


def labels():
    d=pd.read_csv(ROOT/'data/atlas/schaefer100_labels.txt',sep=r'\s+',header=None,names=['id','label','r','g','b','unused'])
    d['hemisphere']=d.label.str.split('_').str[1]
    d['network']=d.label.str.split('_').str[2]
    return d


def endpoints_to_matrix(streamlines, atlas, affine, radius=6.):
    vox=np.argwhere(atlas>0)
    xyz=nib.affines.apply_affine(affine,vox)
    tree=cKDTree(xyz); values=atlas[tuple(vox.T)].astype(int)
    counts=np.zeros((N,N),int); pairs=[];distances=[]
    for sl in streamlines:
        dist,idx=tree.query(np.array([sl[0],sl[-1]]))
        pair=values[idx]-1
        distances.append(dist)
        if np.max(dist)<=radius and pair[0]!=pair[1]:
            i,j=pair;counts[i,j]+=1;counts[j,i]+=1;pairs.append(pair)
        else:pairs.append([-1,-1])
    return counts,np.array(pairs),np.array(distances)


def structural(session, seeds_count=30000, variant='corrected'):
    if variant not in ('corrected','uncorrected_matched'):raise ValueError(variant)
    out=RESULTS/session if variant=='corrected' else RESULTS/'uncorrected_matched'/session
    if (out/'sc_qc.json').exists():
        previous=json.loads((out/'sc_qc.json').read_text())
        if previous.get('method_version')=='shell-specific-v2' and previous.get('preprocessing')==variant and previous.get('seeds')==seeds_count:return
    start=time.time()
    if variant=='corrected' and not (out/'dwi_correction.json').exists():
        raise RuntimeError('Run FSL topup/eddy correction before structural reconstruction')
    img=nib.load(out/('dwi_corrected.nii.gz' if variant=='corrected' else 'dwi_motion_corrected.nii.gz'));data=img.get_fdata(dtype=np.float32)
    bval=np.loadtxt(out/'dwi.bval');bvec=np.loadtxt(out/'dwi.bvec').T
    # Acquired protocol is multi-shell. DTI uses b0 + b<=1200; single-shell
    # CSD uses b0 + b=2800 only, rather than mixing b-dependent responses.
    tensor_use=bval<=1200
    csd_use=(bval<50)|(np.abs(bval-2800)<100)
    tensor_gtab=gradient_table(bval[tensor_use],bvecs=bvec[tensor_use],b0_threshold=50)
    csd_gtab=gradient_table(bval[csd_use],bvecs=bvec[csd_use],b0_threshold=50)
    brain=nib.load(out/'brain_mask_dwi.nii.gz').get_fdata()>.5
    invalid_geometry=0
    if variant=='corrected':
        # Nonpositive volume changes cannot support a reliable local fit.
        ap_jac=nib.load(out/'correction/jacobian_01.nii.gz').get_fdata()
        pa_jac=nib.load(out/'correction/jacobian_07.nii.gz').get_fdata()
        geometry=(ap_jac>0)&(pa_jac>0)&np.isfinite(ap_jac)&np.isfinite(pa_jac)
        invalid_geometry=int((brain&~geometry).sum());brain &= geometry
    b0=data[...,bval<50].mean(-1)
    brain &= b0>np.percentile(b0[brain],10)
    nib.save(nib.Nifti1Image(brain.astype('uint8'),img.affine),out/'structural_fit_mask.nii.gz')
    print(session,'tensor fit',int(brain.sum()),flush=True)
    fit=TensorModel(tensor_gtab,fit_method='WLS').fit(data[...,tensor_use],mask=brain)
    fa=np.nan_to_num(fractional_anisotropy(fit.evals)).astype('float32')
    fa=np.clip(fa,0,1);md=fit.md.astype('float32')
    for name,x in [('fa',fa),('md',md)]:nib.save(nib.Nifti1Image(x,img.affine),out/f'{name}.nii.gz')
    # Whole-brain high-FA response; explicit measured response, not an arbitrary template.
    wm=(fa>.7)&brain&(md>0)&(md<.002)
    if wm.sum()<50:raise RuntimeError('Insufficient high-FA response voxels: inspect DWI and b-vectors')
    high_fit=TensorModel(csd_gtab,fit_method='WLS').fit(data[...,csd_use],mask=wm)
    response=(np.median(high_fit.evals[wm],axis=0),float(np.median(b0[wm])))
    print(session,'CSD fit',response[0],flush=True)
    model=ConstrainedSphericalDeconvModel(csd_gtab,response,sh_order_max=6)
    csd=model.fit(data[...,csd_use],mask=brain)
    np.save(out/'csd_sh_coeff.npy',csd.shm_coeff.astype('float32'))
    getter=DeterministicMaximumDirectionGetter.from_shcoeff(csd.shm_coeff,max_angle=35.,sphere=default_sphere,pmf_threshold=.1)
    seeds=random_seeds_from_mask(brain&(fa>.25),img.affine,seeds_count=seeds_count,seed_count_per_voxel=False,random_seed=160)
    tracker=LocalTracking(getter,ThresholdStoppingCriterion(fa,.1),seeds,img.affine,step_size=.75,max_cross=1,return_all=True,random_seed=160,maxlen=400)
    streamlines=[s for s in tracker if 20<=length(s)<=250]
    print(session,'streamlines',len(streamlines),flush=True)
    atlas=nib.load(out/'atlas_dwi.nii.gz').get_fdata().astype(int)
    counts,pairs,dist=endpoints_to_matrix(streamlines,atlas,img.affine)
    np.save(out/'sc_counts.npy',counts);np.save(out/'sc_fraction.npy',counts/max(counts.sum()/2,1))
    np.save(out/'sc_per_10000_seeds.npy',counts/seeds_count*10000)
    np.savez_compressed(out/'streamline_endpoints.npz',pairs=pairs,distances_mm=dist,lengths_mm=np.array([length(s) for s in streamlines]))
    tractogram=nib.streamlines.Tractogram(streamlines,affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram,str(out/'tractogram.tck'))
    # Alternative endpoint radii quantify a major construction choice.
    for radius in [2.,4.,8.]:
        c,_,_=endpoints_to_matrix(streamlines,atlas,img.affine,radius)
        np.save(out/f'sc_radius_{int(radius)}mm.npy',c)
    (out/'sc_qc.json').write_text(json.dumps({'status':'exploratory patient-derived','method_version':'shell-specific-v2','preprocessing':variant,'excluded_anatomical_mask_voxels_invalid_jacobian':invalid_geometry,'model':'b=2800 single-tissue CSD, SH order 6; deterministic tracking','tensor_bvalues_s_mm2':np.unique(bval[tensor_use]).tolist(),'csd_bvalues_s_mm2':np.unique(bval[csd_use]).tolist(),'csd_volumes':int(csd_use.sum()),'seeds':seeds_count,'random_seed':160,'length_filtered_streamlines':len(streamlines),'assigned_interparcel_streamlines':int(counts.sum()/2),'response_voxels':int(wm.sum()),'response_eigenvalues_mm2_s':response[0].tolist(),'median_wm_fa':float(np.median(fa[brain&(fa>.25)])),'tracking_step_mm':.75,'FA_stop':.1,'max_angle_deg':35,'min_length_mm':20,'max_length_mm':250,'endpoint_radius_mm':6,'seconds':time.time()-start,'limitations':(['rigid motion only; no susceptibility/eddy correction'] if variant!='corrected' else [])+['no ACT or SIFT2','counts are not axon counts','affine atlas plus rigid longitudinal registration','no subcortical or cerebellar nodes']},indent=2))


def fd_from_motion(motion, radius=50.):
    delta=np.diff(motion,axis=0,prepend=motion[:1])
    return np.abs(delta[:,3:]).sum(1)+radius*np.abs(delta[:,:3]).sum(1)


def clean_timeseries(y,motion,noncortical_pcs,fd_threshold=.5,global_signal=None,tr=2.4,drop=4,low_pass=None,keep_override=None):
    """Joint stop-band DCT + nuisance regression fitted on retained frames.

    Excluded frames retain their temporal positions; no filtering across a
    concatenated censored series. Rank and residual degrees of freedom saved.
    keep_override supports matched-subset diagnostics on that original axis;
    callers must ensure it is a subset of their eligible censor pool.
    """
    nt=len(y);t=np.arange(nt);fd=fd_from_motion(motion)
    bad=fd>fd_threshold
    bad |= np.r_[False,bad[:-1]] # one following frame
    keep=~bad;keep[:drop]=False
    if keep_override is not None:
        keep=np.asarray(keep_override)
        if keep.dtype!=bool or keep.shape!=(nt,) or keep[:drop].any():
            raise ValueError('Explicit frame mask must be boolean, match original time axis and exclude initial frames')
        keep=keep.copy()
    dm=np.diff(motion,axis=0,prepend=motion[:1])
    conf=np.column_stack([motion,dm,motion**2,dm**2,noncortical_pcs])
    freq=np.arange(1,nt)/(2*nt*tr)
    ks=np.arange(1,nt)[(freq<.01) | ((freq>low_pass) if low_pass is not None else False)]
    stopband=np.cos(np.pi*(t[:,None]+.5)*ks[None,:]/nt)
    X=np.column_stack([t,conf,stopband])
    if global_signal is not None:X=np.column_stack([X,global_signal])
    X=X-X[keep].mean(0);sd=X[keep].std(0);X=X[:,sd>1e-8]/sd[sd>1e-8]
    X=np.column_stack([np.ones(nt),X])
    rank=int(np.linalg.matrix_rank(X[keep]));dof=int(keep.sum()-rank)
    if keep.sum()<60 or dof<15:raise RuntimeError(f'Insufficient FC data: retained={keep.sum()}, residual df={dof}')
    valid=np.isfinite(y).all(0)&(y[keep].std(0)>1e-8)
    cleaned=np.full_like(y,np.nan,dtype=float)
    beta=lstsq(X[keep],y[keep][:,valid],cond=1e-8)[0]
    cleaned[:,valid]=y[:,valid]-X@beta
    fc=np.full((y.shape[1],y.shape[1]),np.nan)
    fc[np.ix_(valid,valid)]=np.corrcoef(cleaned[keep][:,valid],rowvar=False)
    np.fill_diagonal(fc,0.)
    return fc,cleaned,keep,{'retained_frames':int(keep.sum()),'retained_minutes':float(keep.sum()*tr/60),'design_rank':rank,'residual_df':dof,'mean_fd_mm':float(fd[drop:].mean()),'max_fd_mm':float(fd.max()),'fd_threshold_mm':fd_threshold,'valid_rois':int(valid.sum()),'global_signal_regression':global_signal is not None},fd


def _functional_legacy(session, variant='corrected'):
    from .bold import variant_folder, METHOD, sha
    out=variant_folder(session,variant)
    if (out/'fc_qc.json').exists():
        old=json.loads((out/'fc_qc.json').read_text())
        if variant=='uncorrected' or (old.get('method_version')==METHOD and old.get('preprocessing')==variant):return
    if variant!='uncorrected' and not (RESULTS/session/'bold_sdc.json').exists():
        raise RuntimeError('Run BOLD susceptibility correction before FC')
    start=time.time();image_path=out/('func_motion_corrected.nii.gz' if variant=='uncorrected' else 'func_sdc.nii.gz')
    img=nib.load(image_path);data=img.get_fdata(dtype=np.float32)
    atlas=nib.load(out/'atlas_func.nii.gz').get_fdata().astype(int)
    brain=nib.load(out/'brain_mask_func.nii.gz').get_fdata()>.5
    if variant!='uncorrected':brain &= nib.load(out/'bold_valid_mask.nii.gz').get_fdata()>.5
    motion=np.loadtxt(out/'func_motion.tsv')
    ref=read(out/'func_reference.nii.gz');tc=sitk.ReadTransform(str(out/'t1_to_func_pull.tfm'))
    lesion_name='preop_lesion' if session=='preop' else 'postop_preop_lesion_footprint'
    lesion=warp(read(RESULTS/f'anatomy/{lesion_name}.nii.gz'),ref,tc.GetInverse(),True)
    lesion_mask=array(lesion)>.5;save(lesion,out/'preop_lesion_footprint_func.nii.gz')
    mean=data[...,4:].mean(-1);sd=data[...,4:].std(-1);tsnr=mean/np.maximum(sd,1e-5)
    valid_vox=brain&(tsnr>5)&(mean>np.percentile(mean[brain],5))&~lesion_mask
    series=np.full((data.shape[-1],N),np.nan);coverage=[]
    for i in range(1,N+1):
        allvox=atlas==i;vox=allvox&valid_vox
        coverage.append({'id':i,'atlas_voxels':int(allvox.sum()),'usable_voxels':int(vox.sum()),'usable_fraction':float(vox.sum()/max(allvox.sum(),1))})
        if vox.sum()>=8:series[:,i-1]=data[vox].mean(0)
    # Noncortical PCA nuisance components, explicitly NOT claimed to be aCompCor.
    nuisance_mask=ndi.binary_erosion(brain,iterations=2)&~ndi.binary_dilation(atlas>0,iterations=1)&~ndi.binary_dilation(lesion_mask,iterations=1)
    noise=data[nuisance_mask].T.astype(float)
    nsd=noise.std(0);noise=noise[:,nsd>1e-5];noise-=noise.mean(0);noise/=noise.std(0)
    U,S,_=np.linalg.svd(noise,full_matrices=False);pcs=U[:,:5]*S[:5]
    pd.DataFrame(coverage).to_csv(out/'fc_roi_coverage.csv',index=False)
    np.save(out/'roi_timeseries_raw.npy',series);np.save(out/'nuisance_pcs.npy',pcs)
    global_signal=data[brain&~lesion_mask].mean(0)
    qc=[]
    for threshold,gs,name,lowpass in [(.5,None,'primary',None),(.3,None,'fd03',None),(1.,None,'fd10',None),(.5,global_signal,'gsr',None),(.5,None,'bandpass',.1)]:
        try:
            fc,resid,keep,q,fd=clean_timeseries(series,motion,pcs,threshold,gs,low_pass=lowpass)
        except RuntimeError as e:
            if name=='primary':raise
            # A rejected rerun must not leave a previously eligible matrix.
            (out/f'fc_{name}.npy').unlink(missing_ok=True)
            (out/f'fc_{name}_keep.npy').unlink(missing_ok=True)
            qc.append({'analysis':name,'status':str(e)});continue
        np.save(out/f'fc_{name}.npy',fc);np.save(out/f'fc_{name}_keep.npy',keep)
        if name=='primary':np.save(out/'roi_timeseries_clean.npy',resid);np.save(out/'fd.npy',fd)
        q['analysis']=name;q['low_pass_Hz']=lowpass;qc.append(q)
    (out/'fc_qc.json').write_text(json.dumps({'status':'exploratory patient-derived','method_version':METHOD,'preprocessing':variant,'image_sha256':sha(image_path),'analyses':qc,'seconds':time.time()-start,'TR_seconds':2.4,'first_frames_dropped':4,'passband_Hz':[.01,None],'nuisance':'24 motion + 5 noncortical PCs + linear trend, joint DCT stop-band regression','mask':'exclude transported preop lesion footprint; >=8 usable voxels/ROI, tSNR>5; corrected data require field support, all-volume FOV and positive Jacobians','limitations':(['no susceptibility correction'] if variant=='uncorrected' else ['static cross-acquisition field transfer; uncertain timing and shim compatibility'])+['no slice timing correction','no validated tissue segmentation for nuisance PCA','approximate Euler-based FD at 50mm radius','neurovascular uncoupling may affect BOLD']},indent=2))


def functional(session, variant='corrected'):
    """Primary anatomical model; retain historical noncortical PCs for diagnosis.

    The rigid-only archive is immutable. Its newly matched anatomical analysis
    lives in nuisance_investigation instead of replacing archived derivatives.
    """
    from .bold import variant_folder, METHOD, sha
    from .nuisance import OUT, segment, anatomical_masks, pca_signals
    from .denoising import VERSION
    src=variant_folder(session,variant)
    _functional_legacy(session,variant)
    out=src if variant!='uncorrected' else OUT/session/variant
    out.mkdir(parents=True,exist_ok=True)
    segment(session)
    image_path=src/('func_motion_corrected.nii.gz' if variant=='uncorrected' else 'func_sdc.nii.gz')
    sources=[image_path,src/'roi_timeseries_raw.npy',src/'func_motion.tsv',src/'t1_to_func_pull.tfm',src/'brain_mask_func.nii.gz',src/'preop_lesion_footprint_func.nii.gz']
    if variant!='uncorrected':sources.append(src/'bold_valid_mask.nii.gz')
    sources += [OUT/session/f'tissue_pve_{i}.nii.gz' for i in range(3)]
    lineage={str(p.relative_to(ROOT)):sha(p) for p in sources}
    if (out/'fc_qc.json').exists():
        q=json.loads((out/'fc_qc.json').read_text())
        if q.get('denoising_version')==VERSION and q.get('denoising_inputs')==lineage:return out
    start=time.time();data=nib.load(image_path).get_fdata(dtype=np.float32)
    masks,_=anatomical_masks(session,variant,.95);mask=masks['WM']|masks['CSF']
    series=np.load(src/'roi_timeseries_raw.npy');motion=np.loadtxt(src/'func_motion.tsv')
    brain=nib.load(src/'brain_mask_func.nii.gz').get_fdata()>.5
    if variant!='uncorrected':brain &= nib.load(src/'bold_valid_mask.nii.gz').get_fdata()>.5
    lesion=nib.load(src/'preop_lesion_footprint_func.nii.gz').get_fdata()>.5
    global_signal=data[brain&~lesion].mean(0);np.save(out/'primary_global_signal.npy',global_signal)
    qc=[];cache={}
    for threshold,gs,name,lowpass in [(.5,None,'primary',None),(.3,None,'fd03',None),(1.,None,'fd10',None),(.5,global_signal,'gsr',None),(.5,None,'bandpass',.1)]:
        try:
            if threshold not in cache:
                # Derive the censor set before PCA; refit for each FD threshold.
                k=clean_timeseries(series,motion,np.empty((len(series),0)),threshold)[2]
                cache[threshold]=pca_signals(data,mask,k,True,n=5)[0]
            pcs=cache[threshold]
            fc,resid,keep,q,fd=clean_timeseries(series,motion,pcs,threshold,gs,low_pass=lowpass)
        except RuntimeError as e:
            if name=='primary':raise
            for suffix in ['', '_keep']:(out/f'fc_{name}{suffix}.npy').unlink(missing_ok=True)
            qc.append({'analysis':name,'status':str(e)});continue
        np.save(out/f'fc_{name}.npy',fc);np.save(out/f'fc_{name}_keep.npy',keep)
        np.save(out/f'primary_pcs_{name}.npy',pcs)
        if name=='primary':
            np.save(out/'primary_nuisance_pcs.npy',pcs)
            np.save(out/'roi_timeseries_clean.npy',resid);np.save(out/'fd.npy',fd)
        q.update(analysis=name,low_pass_Hz=lowpass,denoising_model='anatomical5',PCA_fit_frames=int(keep.sum()))
        qc.append(q)
    (out/'fc_qc.json').write_text(json.dumps({'status':'exploratory patient-derived','method_version':METHOD,
        'denoising_version':VERSION,'denoising_model':'anatomical5','denoising_inputs':lineage,
        'preprocessing':variant,'image_sha256':sha(image_path),'analyses':qc,'seconds':time.time()-start,
        'TR_seconds':2.4,'first_frames_dropped':4,'passband_Hz':[.01,None],
        'nuisance':'24 motion + 5 combined anatomical WM/CSF PCs (retained-frame temporal preprocessing/scaling/PCA) + trend/DCT',
        'tissue_masks':{'PVE_threshold':.95,'WM_voxels':int(masks['WM'].sum()),'CSF_voxels':int(masks['CSF'].sum())},
        'selection':'Post hoc methodological recommendation; see config/PRIMARY_DENOISING.md',
        'limitations':['Automatic T1 segmentation lacks independent expert validation','Five-PC budget is pragmatic; component-count sensitivity remains',
            'Substantial shared cortical variance removal can include neural signal','No physiological recordings',
            'Static cross-acquisition susceptibility correction is conditional','No slice timing correction','Approximate Euler-based FD at 50 mm',
            'Neurovascular uncoupling may affect BOLD']},indent=2))
    return out


def run():
    from . import SESSIONS
    for s in SESSIONS:
        structural(s);functional(s)
