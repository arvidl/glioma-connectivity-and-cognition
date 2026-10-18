"""BIDS-driven FSL topup/eddy on raw AP data, with auditable CPU execution.

Do not realign the DWI before eddy or rotate its gradients a second time.
AP is the reference grid. As in fslmerge, reversed-PE voxel arrays are stacked
without interpolation; small prescription differences are recorded for QC.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import time

import nibabel as nib
import numpy as np
from dipy.segment.mask import median_otsu
from scipy import ndimage as ndi
from . import DATA, ROOT, RESULTS


def acquisition_row(metadata):
    pe = metadata['PhaseEncodingDirection']
    if pe not in ('i', 'i-', 'j', 'j-', 'k', 'k-'):
        raise ValueError(f'Unsupported BIDS PhaseEncodingDirection: {pe}')
    trt = float(metadata['TotalReadoutTime'])
    if not 0 < trt < 1:
        raise ValueError('TotalReadoutTime must be seconds in (0,1)')
    row = np.zeros(4)
    row['ijk'.index(pe[0])] = -1 if pe.endswith('-') else 1
    row[3] = trt
    return row


def fsl_environment():
    directory = Path(os.environ.get('FSLDIR', Path.home()/'fsl'))
    bindir = directory/'share/fsl/bin'
    if not (bindir/'eddy_cpu').exists():
        candidate = shutil.which('eddy_cpu')
        if not candidate:
            raise RuntimeError('FSL topup and eddy_cpu are required. Set FSLDIR; see README.')
        bindir = Path(candidate).parent
    env = os.environ.copy()
    env.update(FSLDIR=str(directory), FSLOUTPUTTYPE='NIFTI_GZ', OMP_NUM_THREADS='8')
    env['PATH'] = str(bindir)+os.pathsep+env.get('PATH','')
    return directory, bindir, env


def execute(command, folder, env, name):
    (folder/f'{name}_command.json').write_text(json.dumps(command, indent=2))
    print('Running', name, 'in', folder, flush=True)
    start = time.time()
    with (folder/f'{name}.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.time()-start


def correct_diffusion(session):
    from .imaging import coregister
    out = RESULTS/session
    out.mkdir(exist_ok=True, parents=True)
    if (out/'dwi_correction.json').exists():
        return
    folder = out/'correction'
    folder.mkdir(exist_ok=True)
    fsldir, bindir, env = fsl_environment()
    config = ROOT/'config/b02b0.cnf'
    if not config.exists():
        raise RuntimeError('Missing bundled topup configuration')
    stems = {a:DATA/f'sub-PAT16/ses-{session}/dwi/sub-PAT16_ses-{session}_acq-{a}_dwi' for a in ['AP','PA']}
    images = {a:nib.load(str(p)+'.nii.gz') for a,p in stems.items()}
    bvals = {a:np.loadtxt(str(p)+'.bval') for a,p in stems.items()}
    metadata = {a:json.loads(p.with_suffix('.json').read_text()) for a,p in stems.items()}
    rows = {a:acquisition_row(j) for a,j in metadata.items()}
    if not np.array_equal(rows['AP'][:3], -rows['PA'][:3]):
        raise ValueError('This workflow requires opposing AP/PA encodings')
    ap = images['AP']
    if ap.shape[:3] != images['PA'].shape[:3]:
        raise ValueError('AP/PA array grids differ; explicit resampling strategy required')
    axes = {a:im.affine[:3,:3]/np.linalg.norm(im.affine[:3,:3],axis=0) for a,im in images.items()}
    angle = float(np.degrees(np.arccos(np.clip((np.trace(axes['AP'].T@axes['PA'])-1)/2,-1,1))))
    if angle > 1. or not np.allclose(ap.header.get_zooms()[:3],images['PA'].header.get_zooms()[:3],atol=1e-3):
        raise ValueError('AP/PA prescription differs substantially; review geometry before topup')
    bzero = {a:np.flatnonzero(b<50) for a,b in bvals.items()}
    if not all(len(x) for x in bzero.values()) or bzero['AP'][0] != 0:
        raise ValueError('Need first AP b0 and at least one reverse-phase b0')
    stack = np.concatenate([images[a].get_fdata(dtype=np.float32)[...,bzero[a]] for a in ['AP','PA']],axis=3)
    acqp = np.vstack([np.tile(rows[a],(len(bzero[a]),1)) for a in ['AP','PA']])
    nib.save(nib.Nifti1Image(stack,ap.affine,ap.header),folder/'b0_AP_PA.nii.gz')
    np.savetxt(folder/'acqparams.txt',acqp,fmt='%.9g')
    np.savetxt(folder/'index.txt',np.ones((1,ap.shape[3]),int),fmt='%d')
    seconds = {}
    topup = folder/'topup'
    if not (folder/'topup_complete.json').exists():
        seconds['topup'] = execute([str(bindir/'topup'),f'--imain={folder}/b0_AP_PA.nii.gz',
            f'--datain={folder}/acqparams.txt',f'--config={config}',f'--out={topup}',
            f'--iout={folder}/b0_unwarped',f'--fout={folder}/field_Hz',
            f'--dfout={folder}/displacement',f'--jacout={folder}/jacobian','--verbose'],folder,env,'topup')
        (folder/'topup_complete.json').write_text(json.dumps(seconds))
    else:
        seconds.update(json.loads((folder/'topup_complete.json').read_text()))
    b0im = nib.load(folder/'b0_unwarped.nii.gz')
    mean = b0im.get_fdata(dtype=np.float32).mean(3)
    _,mask = median_otsu(mean,median_radius=3,numpass=3,dilate=1)
    mask = ndi.binary_fill_holes(mask)
    labs,_ = ndi.label(mask)
    mask = labs == np.argmax(np.bincount(labs.ravel())[1:])+1
    nib.save(nib.Nifti1Image(mask.astype('uint8'),ap.affine),folder/'brain_mask.nii.gz')
    corrected = out/'dwi_corrected'
    if not (folder/'eddy_complete.json').exists():
        seconds['eddy'] = execute([str(bindir/'eddy_cpu'),f'--imain={stems["AP"]}.nii.gz',
        f'--mask={folder}/brain_mask.nii.gz',f'--acqp={folder}/acqparams.txt',f'--index={folder}/index.txt',
        f'--bvals={stems["AP"]}.bval',f'--bvecs={stems["AP"]}.bvec',f'--topup={topup}',f'--out={corrected}',
        '--flm=quadratic','--slm=none','--niter=5','--fwhm=0','--repol',
        '--cnr_maps','--residuals','--initrand=160','--nthr=8','--verbose'],folder,env,'eddy')
        (folder/'eddy_complete.json').write_text(json.dumps({'eddy':seconds['eddy']}))
    else:
        seconds.update(json.loads((folder/'eddy_complete.json').read_text()))
    shutil.copy2(str(stems['AP'])+'.bval',out/'dwi.bval')
    shutil.copy2(out/'dwi_corrected.eddy_rotated_bvecs',out/'dwi.bvec')
    result = nib.load(out/'dwi_corrected.nii.gz')
    data = result.get_fdata(dtype=np.float32)
    if data.shape != ap.shape or not np.isfinite(data).all():
        raise RuntimeError('Invalid corrected DWI array')
    nib.save(nib.Nifti1Image(data[...,bzero['AP']].mean(3),result.affine),out/'dwi_reference.nii.gz')
    registration = coregister('dwi',session)
    field = nib.load(folder/'field_Hz.nii.gz').get_fdata()
    movement = np.loadtxt(out/'dwi_corrected.eddy_movement_rms')
    outliers = np.loadtxt(out/'dwi_corrected.eddy_outlier_map',skiprows=1)
    # Same intensity scaling for both polarities, before/after; descriptive QC.
    n_ap = len(bzero['AP'])
    after = b0im.get_fdata(dtype=np.float32)
    def concordance(d):
        x=d[...,:n_ap].mean(3)[mask];y=d[...,n_ap:].mean(3)[mask]
        return float(np.corrcoef(x,y)[0,1])
    help_output=subprocess.run([str(bindir/'eddy_cpu'),'--help'],env=env,capture_output=True,text=True)
    version_match=re.search(r'Part of FSL \(ID: (.*?)\)',help_output.stdout+help_output.stderr)
    qc = {'method':'FSL topup + eddy_cpu', 'fsl_version':(fsldir/'etc/fslversion').read_text().strip(),
        'binary_id':version_match.group(1) if version_match else 'unavailable',
        'seconds':seconds,'frames':data.shape[3],'b0_indices':{a:x.tolist() for a,x in bzero.items()},
        'shells_s_mm2':{str(int(b)):int((bvals['AP']==b).sum()) for b in np.unique(bvals['AP'])},
        'phase_encoding':{a:metadata[a]['PhaseEncodingDirection'] for a in metadata},
        'total_readout_seconds':{a:metadata[a]['TotalReadoutTime'] for a in metadata},
        'grid_handling':'AP header; AP then PA voxel arrays stacked without pre-registration or interpolation (fslmerge convention)',
        'AP_PA_prescription_angle_deg':angle,'AP_PA_origin_difference_mm':(images['PA'].affine[:3,3]-ap.affine[:3,3]).tolist(),
        'topup_config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
        'field_Hz_brain_percentiles_1_50_99':np.percentile(field[mask],[1,50,99]).tolist(),
        'AP_displacement_vox_brain_percentiles_1_50_99':np.percentile(field[mask]*rows['AP'][3]*rows['AP'][:3].sum(),[1,50,99]).tolist(),
        'b0_AP_PA_correlation_before':concordance(stack),'b0_AP_PA_correlation_after':concordance(after),
        'mean_absolute_movement_rms_mm':float(movement[:,0].mean()),'max_absolute_movement_rms_mm':float(movement[:,0].max()),
        'outlier_slices':int(outliers.sum()),'outlier_slice_entries':int(outliers.size),'outlier_slice_fraction':float(outliers.mean()),
        'registration':registration,'gradient_source':'eddy_rotated_bvecs, used directly; no second rigid rotation',
        'corrections':['susceptibility (topup)','eddy currents (quadratic)','volume-to-volume motion','slice outlier replacement'],
        'not_performed':['slice-to-volume motion','susceptibility-by-motion','BOLD susceptibility correction'],
        'limitations':['Small AP/PA prescription differences approximated by topup rigid motion in common voxel grid; review QC',
                       'AP/PA agreement is an internal fit check, not independent anatomical validation']}
    (out/'dwi_correction.json').write_text(json.dumps(qc,indent=2))
    print(session,'DWI correction complete',seconds,flush=True)
