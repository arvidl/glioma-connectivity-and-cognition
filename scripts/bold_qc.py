"""BOLD correction QC, paired FC/coupling sensitivity, and nuisance-PC diagnostic."""
import json
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pat16 import DATA, RESULTS, SESSIONS
from pat16.bold import variant_folder
from pat16.connectivity import functional, clean_timeseries
from pat16.networks import metrics, sc_fc_coupling

primary_folders={}
for session in SESSIONS:
    for variant in ('uncorrected','corrected','readout_alternative'):
        primary_folders[session,variant]=functional(session,variant)
valid = []
for session in SESSIONS:
    for variant in ('uncorrected','corrected','readout_alternative'):
        valid.append(np.isfinite(np.load(variant_folder(session,variant)/'roi_timeseries_raw.npy')).all(0))
common = np.logical_and.reduce(valid)
np.save(RESULTS/'bold_comparison_common_nodes.npy',common)
i,j = np.triu_indices(common.sum(),1)
rows=[];qrows=[];nuisance=[]
fig,axs=plt.subplots(2,4,figsize=(12.6,6.7))
for row,session in enumerate(SESSIONS):
    out=RESULTS/session;before=variant_folder(session,'uncorrected')
    sc=np.load(out/'sc_fraction.npy')[np.ix_(common,common)]
    old=np.load(primary_folders[session,'uncorrected']/'fc_primary.npy')[np.ix_(common,common)]
    q=json.loads((out/'bold_sdc.json').read_text())
    v=q['variants']['corrected']
    qrows.append({'session':session,'BOLD_readout_seconds':q['BOLD_total_readout_seconds'],
        'field_scale_alternative':q['timing_alternative']['displacement_scale_relative_to_primary'],
        'valid_reference_brain_fraction':v['reference_brain_covered_fraction'],
        'nonpositive_jacobian_reference_brain':v['nonpositive_min_jacobian_in_reference_brain'],
        'shift_p01_vox':v['signed_shift_vox_brain_percentiles_1_50_99'][0],
        'shift_median_vox':v['signed_shift_vox_brain_percentiles_1_50_99'][1],
        'shift_p99_vox':v['signed_shift_vox_brain_percentiles_1_50_99'][2]})
    for variant in ('uncorrected','corrected','readout_alternative'):
        p=primary_folders[session,variant];fc=np.load(p/'fc_primary.npy')[np.ix_(common,common)]
        qc=json.loads((p/'fc_qc.json').read_text())['analyses'][0]
        rows.append({'session':session,'BOLD_processing':variant,'denoising_model':'anatomical5','common_nodes':int(common.sum()),
            'mean_fc_r':float(fc[i,j].mean()),'mean_absolute_fc_difference_from_uncorrected':float(abs(fc-old)[i,j].mean()),
            'sc_fc_spearman':sc_fc_coupling(sc,fc),'retained_frames':qc['retained_frames'],
            'residual_df':qc['residual_df'],**metrics(np.maximum(fc,0),.2,1.)})
    # Swap component sets solely as a diagnostic of nuisance sensitivity, not
    # as independent physiological validation or a preferred denoising model.
    for roi_variant in ('uncorrected','corrected'):
        for pc_variant in ('uncorrected','corrected'):
            y=np.load(variant_folder(session,roi_variant)/'roi_timeseries_raw.npy')
            pcs=np.load(variant_folder(session,pc_variant)/'nuisance_pcs.npy')
            f,_,_,_,_=clean_timeseries(y,np.loadtxt(out/'func_motion.tsv'),pcs)
            f=f[np.ix_(common,common)]
            nuisance.append({'session':session,'diagnostic_model':'historical_noncortical5','ROI_processing':roi_variant,'PC_processing':pc_variant,
                'mean_fc_r':float(f[i,j].mean()),'sc_fc_spearman':sc_fc_coupling(sc,f),'common_nodes':int(common.sum())})
    # Identical physical T1 slice for before/after, each using its own saved rigid fit.
    t1=nib.as_closest_canonical(nib.load(DATA/f'sub-PAT16/ses-{session}/anat/sub-PAT16_ses-{session}_T1w.nii.gz')).get_fdata()
    for col,p,label in [(0,before,'Rigid-only BOLD'),(1,out,'Susceptibility-corrected')]:
        x=nib.as_closest_canonical(nib.load(p/'func_in_t1.nii.gz')).get_fdata()
        axs[row,col].imshow(t1[:,:,145].T,origin='lower',cmap='gray',vmin=0,vmax=np.percentile(t1[t1>0],99))
        axs[row,col].contour(x[:,:,145].T,levels=np.percentile(x[x>0],[40,65,80]),colors=['cyan','orange','red'],linewidths=.6)
        axs[row,col].set_title(f'{session}: {label}',fontsize=10);axs[row,col].axis('off')
    mask=nib.as_closest_canonical(nib.load(out/'bold_valid_mask.nii.gz')).get_fdata()>.5
    for col,name,title,lo,hi in [(2,'bold_shift_vox','Signed PE shift (vox)',-2,2),(3,'bold_min_jacobian','Minimum Jacobian',.5,1.5)]:
        x=nib.as_closest_canonical(nib.load(out/f'{name}.nii.gz')).get_fdata()
        im=axs[row,col].imshow(np.where(mask,x,np.nan)[:,:,22].T,origin='lower',cmap='RdBu_r',vmin=lo,vmax=hi)
        axs[row,col].set_title(title,fontsize=10);axs[row,col].axis('off');fig.colorbar(im,ax=axs[row,col],fraction=.04,pad=.02)
fig.suptitle('BOLD static field transfer: anatomical QC and valid correction geometry',fontsize=14)
fig.text(.5,.01,'T1 slice 145; BOLD slice 22. Separate rigid fits; field transfer is conditional on static field/shim compatibility.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.035,1,.95]);fig.savefig(RESULTS/'figures/bold_distortion_qc.png',dpi=180);plt.close(fig)
df=pd.DataFrame(rows);df.to_csv(RESULTS/'bold_comparison.csv',index=False)
pd.DataFrame(qrows).to_csv(RESULTS/'bold_correction_qc.csv',index=False)
nd=pd.DataFrame(nuisance);nd.to_csv(RESULTS/'bold_nuisance_sensitivity.csv',index=False)
fig,axs=plt.subplots(1,3,figsize=(11,3.7))
for ax,column,title in zip(axs,['mean_fc_r','sc_fc_spearman','weighted_global_efficiency'],['Mean FC (Pearson r)','SC–FC coupling (Spearman)','Positive-FC weighted efficiency']):
    for variant,color,label in [('uncorrected','#999999','Rigid-only + anatomical 5'),('corrected','#328c91','Static transfer + anatomical 5'),('readout_alternative','#a777b4','Alternative timing + anatomical 5')]:
        d=df[df.BOLD_processing==variant].set_index('session').loc[list(SESSIONS)]
        ax.plot(['Pre-op','Post-op'],d[column],marker='o',color=color,label=label)
    ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False)
axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(RESULTS/'figures/bold_comparison.png',dpi=180);plt.close(fig)
print(df.to_string(index=False));print('Nuisance component sensitivity:');print(nd.to_string(index=False))
