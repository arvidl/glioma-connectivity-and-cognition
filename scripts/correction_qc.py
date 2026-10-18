"""QC of fitted field, b0 agreement, motion and replaced slices."""
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pat16 import ROOT,DATA,RESULTS,SESSIONS

fig,axs=plt.subplots(2,5,figsize=(13,5.8))
motionfig,maxs=plt.subplots(2,2,figsize=(11,5.6))
rows=[]
for row,s in enumerate(SESSIONS):
    out=RESULTS/s;f=out/'correction'
    q=json.loads((out/'dwi_correction.json').read_text())
    if not q.get('binary_id'):q['binary_id']=json.loads((ROOT/'config/fsl_environment.json').read_text())['binary_build_id']
    q['readout_metadata_audit']={}
    for polarity in ['AP','PA']:
        meta=json.loads((DATA/f'sub-PAT16/ses-{s}/dwi/sub-PAT16_ses-{s}_acq-{polarity}_dwi.json').read_text())
        candidate=meta['EffectiveEchoSpacing']*(meta['PhaseEncodingLines']-1)
        q['readout_metadata_audit'][polarity]={'supplied_seconds':meta['TotalReadoutTime'],'echo_spacing_times_phase_lines_minus_one_seconds':candidate,'consistent':bool(np.isclose(candidate,meta['TotalReadoutTime'],rtol=.01)), 'decision':'Use supplied same TRT consistently in topup/eddy; displacement identifiable, physical Hz scale unverified; no BOLD transfer'}
    native_mask=nib.load(f/'brain_mask.nii.gz').get_fdata()>.5
    field_native=nib.load(f/'field_Hz.nii.gz').get_fdata()
    shift=field_native[native_mask]*q['total_readout_seconds']['AP']
    q['AP_displacement_vox_brain_percentiles_1_50_99']=np.percentile(-shift,[1,50,99]).tolist()
    q['absolute_displacement_vox_brain_percentiles_50_95_99']=np.percentile(abs(shift),[50,95,99]).tolist()
    q['displacement_definition']='Forward phase-encoding displacement = field Hz * supplied readout seconds * PE sign (AP j-), in voxel units'
    for name,suffix in [('AP','01'),('PA','07')]:
        jac=nib.load(f/f'jacobian_{suffix}.nii.gz').get_fdata()
        q[f'{name}_nonpositive_jacobian_brain_voxels']=int((native_mask&(jac<=0)).sum())
        q[f'{name}_jacobian_brain_percentiles_0_1_50_99_100']=np.percentile(jac[native_mask],[0,1,50,99,100]).tolist()
    (out/'dwi_correction.json').write_text(json.dumps(q,indent=2))
    rows.append({'session':s,**{k:q[k] for k in ['b0_AP_PA_correlation_before','b0_AP_PA_correlation_after','mean_absolute_movement_rms_mm','max_absolute_movement_rms_mm','outlier_slices','outlier_slice_fraction','AP_PA_prescription_angle_deg','AP_nonpositive_jacobian_brain_voxels','PA_nonpositive_jacobian_brain_voxels']}})
    a=nib.as_closest_canonical(nib.load(f/'b0_AP_PA.nii.gz')).get_fdata()
    b=nib.as_closest_canonical(nib.load(f/'b0_unwarped.nii.gz')).get_fdata()
    field=nib.as_closest_canonical(nib.load(f/'field_Hz.nii.gz')).get_fdata()
    mask=nib.as_closest_canonical(nib.load(f/'brain_mask.nii.gz')).get_fdata()>.5
    k=28;vmax=np.percentile(a[a>0],98)
    for ax,x,title in zip(axs[row],[a[...,:6].mean(3),a[...,6:].mean(3),b[...,:6].mean(3),b[...,6:].mean(3),np.where(mask,field,np.nan)],['Raw AP','Raw PA','Topup AP','Topup PA','Field (Hz)']):
        if title.endswith('(Hz)'):im=ax.imshow(x[:,:,k].T,origin='lower',cmap='RdBu_r',vmin=-150,vmax=150);fig.colorbar(im,ax=ax,fraction=.04,pad=.02)
        else:ax.imshow(x[:,:,k].T,origin='lower',cmap='gray',vmin=0,vmax=vmax)
        ax.set_title(title,fontsize=14);ax.axis('off')
        if title=='Raw AP':ax.text(-.07,.5,s,rotation=90,transform=ax.transAxes,ha='right',va='center',fontsize=14)
    rms=np.loadtxt(out/'dwi_corrected.eddy_movement_rms');p=np.loadtxt(out/'dwi_corrected.eddy_outlier_map',skiprows=1)
    ax=maxs[row,0];ax.plot(rms[:,0],label='Absolute');ax.plot(rms[:,1],label='Relative');ax.set_ylabel(f'{s}: RMS (mm)');ax.set_xlabel('AP volume');ax.legend(fontsize=8)
    ax=maxs[row,1];ax.imshow(p.T,aspect='auto',origin='lower',vmin=0,vmax=1,cmap='Greys');ax.set_title(f'{s}: eddy flagged slices ({int(p.sum())}/{p.size})',fontsize=10);ax.set_xlabel('AP volume');ax.set_ylabel('Slice')
fig.suptitle('DWI AP/PA correction: common intensity scale within each session',fontsize=15)
fig.text(.5,.01,'Canonical axial slice 28; internal agreement is a fit check. Field may be unreliable at signal-loss boundaries.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.035,1,.95]);fig.savefig(RESULTS/'figures/dwi_distortion_qc.png',dpi=180);plt.close(fig)
motionfig.tight_layout();motionfig.savefig(RESULTS/'figures/dwi_motion_outliers.png',dpi=180);plt.close(motionfig)
pd.DataFrame(rows).to_csv(RESULTS/'dwi_correction_qc.csv',index=False)
