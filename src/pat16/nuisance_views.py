"""Offline visual checks and interactive views for the nuisance investigation."""
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from . import RESULTS,DATA
from .nuisance import OUT
from .imaging import read,warp,array


def canonical(image):
    return nib.as_closest_canonical(image).get_fdata()


def show_masks(session='postop',z=145):
    p=RESULTS/session
    img=nib.load(DATA/f'sub-PAT16/ses-{session}/anat/sub-PAT16_ses-{session}_T1w.nii.gz')
    t1=canonical(img);fixed=read(DATA/f'sub-PAT16/ses-{session}/anat/sub-PAT16_ses-{session}_T1w.nii.gz')
    pull=sitk.ReadTransform(str(p/'t1_to_func_pull.tfm'))
    def native(name):return canonical(nib.load(OUT/session/name))
    def in_t1(name):
        a=array(warp(read(OUT/session/'corrected'/name),fixed,pull,True))
        return canonical(nib.Nifti1Image(a,img.affine))
    wm=native('wm_T1_core_p95.nii.gz')>.5;csf=native('csf_T1_core_p95.nii.gz')>.5
    masks=[[(wm,'#e09b38'),(csf,'#27c8e6')],[(in_t1('original_noncortical_mask.nii.gz')>.5,'#bb62bc')],
           [(in_t1('wm_mask_p95.nii.gz')>.5,'#e09b38'),(in_t1('csf_mask_p95.nii.gz')>.5,'#27c8e6')]]
    fig,axs=plt.subplots(1,3,figsize=(11,4))
    for ax,mm,title in zip(axs,masks,['Native eroded T1 WM/CSF','Original noncortical EPI mask','Anatomical EPI WM/CSF masks']):
        ax.imshow(t1[:,:,z].T,origin='lower',cmap='gray',vmin=0,vmax=np.percentile(t1[t1>0],99))
        for mask,color in mm:
            overlay=np.zeros(mask.shape[:2]+(4,))
            from matplotlib.colors import to_rgba
            overlay[mask[:,:,z]]=to_rgba(color,.65)
            ax.imshow(overlay.transpose(1,0,2),origin='lower')
        ax.set_title(title,fontsize=10);ax.axis('off');ax.set_ylim(35,230)
    fig.suptitle(f'{session}: mask QC on canonical T1 slice {z} | WM orange; CSF cyan; original magenta',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.92]);return fig


def show_model(session='postop',model='anatomical5'):
    f=np.load(OUT/session/'corrected'/f'fc_{model}.npy')
    old=np.load(RESULTS/session/'fc_primary.npy')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,a,title in zip(axs,[f,f-old],[model,'Difference from primary anatomical 5']):
        im=ax.imshow(a,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_title(title,fontsize=10);ax.set_xlabel('Parcel');fig.colorbar(im,ax=ax)
    fig.tight_layout();return fig


def show_cutoff(session='postop',components=5):
    d=pd.read_csv(RESULTS/'nuisance_pc_cutoff.csv');d=d[d.session==session]
    fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    for source,c in [('uncorrected','#999999'),('corrected','#328c91')]:
        a=d[d.PC_source==source]
        axs[0].plot(a.components,a.mean_fc_r,'o-',label=source,color=c,ms=3)
        axs[1].plot(a.components,a.cortical_mean_partial_R2,'o-',label=source,color=c,ms=3)
    for ax in axs:ax.axvline(components,color='crimson',ls=':');ax.set_xlabel('Number of PCs')
    axs[0].set_ylabel('Mean FC r');axs[1].set_ylabel('Cortical-mean partial R²');axs[0].legend();fig.tight_layout()
    from IPython.display import display
    display(d[d.components==components][['PC_source','mean_fc_r','sc_fc_spearman','cortical_mean_partial_R2','residual_df']])
    return fig
