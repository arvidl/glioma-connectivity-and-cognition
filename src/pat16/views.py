"""Interactive views; all assets and data are local."""
from functools import lru_cache
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from IPython.display import display,HTML
from . import DATA,RESULTS
from .networks import proportional_threshold
from .primary_sc import load_weights


@lru_cache(maxsize=12)
def volume(path):return nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)


def show_slice(session='preop',z=140,overlay='atlas'):
    path=DATA/f'sub-PAT16/ses-{session}/anat/sub-PAT16_ses-{session}_T1w.nii.gz'
    a=volume(str(path));fig,ax=plt.subplots(figsize=(6,6))
    ax.imshow(a[:,:,z].T,origin='lower',cmap='gray',vmax=np.percentile(a[a>0],99))
    if overlay=='atlas':
        b=volume(str(RESULTS/f'anatomy/{session}_atlas.nii.gz'));ax.imshow(np.ma.masked_equal(b[:,:,z].T,0),origin='lower',cmap='turbo',vmin=1,vmax=100,alpha=.4)
    elif overlay in ['dwi','func']:
        b=volume(str(RESULTS/f'{session}/{overlay}_in_t1.nii.gz'));ax.contour(b[:,:,z].T,levels=np.percentile(b[b>0],[35,60,80]),colors=['cyan','orange','red'],linewidths=.7)
    else:
        name='preop_lesion' if session=='preop' else 'postop_preop_lesion_footprint'
        b=volume(str(RESULTS/f'anatomy/{name}.nii.gz'));ax.imshow(np.ma.masked_equal(b[:,:,z].T,0),origin='lower',cmap='autumn',alpha=.65)
    ax.set_title(f'{session} | canonical voxel slice {z} | {overlay}');ax.axis('off');plt.show()


def connectome3d(session='preop',modality='SC',density=.03):
    nodes=pd.read_csv(RESULTS/'nodes.csv');xyz=nodes[['x','y','z']].to_numpy()
    w=load_weights(session) if modality=='SC' else np.load(RESULTS/session/'fc_primary.npy')
    w=proportional_threshold(np.maximum(w,0),density)
    i,j=np.where(np.triu(w,1)>0)
    points=np.stack([xyz[i],xyz[j],np.full_like(xyz[i],np.nan)],axis=1).reshape(-1,3)
    fig=go.Figure(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='lines',line=dict(color='rgba(60,80,100,.25)',width=2),hoverinfo='skip',name='Retained positive edges'))
    networks=list(dict.fromkeys(nodes.network));colors=['#7759a4','#3d7da4','#2b9b82','#d4952e','#bb655a','#5f6c9b','#bb709d']
    for net,color in zip(networks,colors):
        mask=nodes.network==net;p=xyz[mask]
        text=[f'{r.label}<br>Pre-op lesion overlap: {r.lesion_overlap_fraction:.1%}<br>Distance: {r.distance_to_lesion_mm:.1f} mm' for _,r in nodes[mask].iterrows()]
        fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker=dict(size=5,color=color),text=text,hovertemplate='%{text}<extra></extra>',name=net))
    l=nib.load(RESULTS/'anatomy/preop_lesion.nii.gz');vox=np.argwhere(l.get_fdata()>.5)[::80];p=nib.affines.apply_affine(l.affine,vox)
    fig.add_trace(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='markers',marker=dict(size=2,color='#e0a23b',opacity=.25),name='Pre-op lesion footprint'))
    fig.update_layout(title=f'{session}: {modality} | requested density {density:.0%}',height=600,scene=dict(aspectmode='data',xaxis_title='R (mm)',yaxis_title='A (mm)',zaxis_title='S (mm)'),margin=dict(l=0,r=0,t=50,b=0))
    return fig


def show_plotly(fig):
    # Inline plotly.js ensures offline operation in Jupyter without a CDN.
    display(HTML(fig.to_html(full_html=False,include_plotlyjs=True)))


def show_tracts(session='preop',n=300):
    t=nib.streamlines.load(str(RESULTS/session/'tractogram.tck'),lazy_load=True)
    points=[]
    for i,sl in enumerate(t.streamlines):
        if i>=n:break
        points.extend([*sl[::3],np.full(3,np.nan)])
    p=np.asarray(points)
    fig=go.Figure(go.Scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode='lines',line=dict(width=2,color='#398a9f'),hoverinfo='skip'))
    fig.update_layout(title=f'{session}: first {n} original seed-160 streamlines (display subset)',height=550,scene=dict(aspectmode='data'))
    show_plotly(fig)


def show_diffusion_correction(session='preop',z=28):
    """Inspect AP/PA b0 agreement and fitted off-resonance in one grid."""
    folder=RESULTS/session/'correction'
    before=volume(str(folder/'b0_AP_PA.nii.gz'))
    after=volume(str(folder/'b0_unwarped.nii.gz'))
    field=volume(str(folder/'field_Hz.nii.gz'))
    mask=volume(str(folder/'brain_mask.nii.gz'))>.5
    fig,axs=plt.subplots(1,5,figsize=(14,4))
    maximum=np.percentile(before[before>0],98)
    for ax,a,title in zip(axs,[before[...,:6].mean(3),before[...,6:].mean(3),after[...,:6].mean(3),after[...,6:].mean(3),np.where(mask,field,np.nan)],['Raw AP','Raw PA','Topup AP','Topup PA','Field (Hz)']):
        isfield=title=='Field (Hz)'
        im=ax.imshow(a[:,:,z].T,origin='lower',cmap='RdBu_r' if isfield else 'gray',vmin=-150 if isfield else 0,vmax=150 if isfield else maximum)
        if isfield:fig.colorbar(im,ax=ax,fraction=.046,pad=.04)
        ax.set_title(title);ax.axis('off')
    fig.suptitle(f'{session}: AP/PA b0 and susceptibility field, canonical slice {z}')
    fig.tight_layout();plt.show()


def show_bold_correction(session='preop',z=22):
    """Compare native BOLD geometry, transferred b0 and static PE displacement."""
    from .imaging import read,warp
    import SimpleITK as sitk
    out=RESULTS/session
    before=volume(str(RESULTS/'bold_uncorrected'/session/'func_reference.nii.gz'))
    after=volume(str(out/'func_reference.nii.gz'))
    transform=sitk.ReadTransform(str(out/'bold_sdc/bold_to_b0_pull.tfm'))
    b0=warp(read(out/'dwi_reference.nii.gz'),read(out/'func_reference.nii.gz'),transform)
    # SITK -> NIfTI keeps the oblique spatial orientation for canonical display.
    from .imaging import array
    b0=nib.as_closest_canonical(nib.Nifti1Image(array(b0),nib.load(out/'func_reference.nii.gz').affine)).get_fdata()
    shift=volume(str(out/'bold_shift_vox.nii.gz'));mask=volume(str(out/'bold_valid_mask.nii.gz'))>.5
    fig,axs=plt.subplots(1,4,figsize=(13,4));vmax=np.percentile(before[before>0],98)
    for ax,x,title in zip(axs,[before,after,b0,np.where(mask,shift,np.nan)],['Rigid-only reference','Corrected reference','Rigidly transferred b0','Signed PE shift (vox)']):
        isshift=title.startswith('Signed')
        im=ax.imshow(x[:,:,z].T,origin='lower',cmap='RdBu_r' if isshift else 'gray',vmin=-2 if isshift else 0,vmax=2 if isshift else (np.percentile(b0[b0>0],98) if title.endswith('b0') else vmax))
        if isshift:fig.colorbar(im,ax=ax,fraction=.045,pad=.02)
        ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle(f'{session}: static field transfer, canonical BOLD slice {z}');fig.tight_layout();plt.show()
