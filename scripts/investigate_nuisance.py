"""Recompute nuisance investigation from saved images and optional cached FAST."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]/".cache/matplotlib"))
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import orth
from pat16 import RESULTS,SESSIONS
from pat16.bold import variant_folder,sha
from pat16.nuisance import OUT,METHOD,segment,legacy_mask,anatomical_masks,pca_signals,save_array,subspace_similarity,base_residuals,cubic_motion_control,linear_sampling_audit
from pat16.connectivity import clean_timeseries,labels
from pat16.networks import sc_fc_coupling,metrics,network_means

p=argparse.ArgumentParser();p.add_argument('--segment-force',action='store_true');args=p.parse_args()
OUT.mkdir(exist_ok=True)
rows=[];maskrows=[];curves=[];factorial=[];diagnostics=[];networkrows=[];components=[];thresholdrows=[];audits=[]
groups=labels().network.to_numpy();common=np.load(RESULTS/'common_fc_nodes.npy');ii,jj=np.triu_indices(common.sum(),1)
for session in SESSIONS:
    print('Investigating',session,flush=True)
    segment(session,args.segment_force)
    audits.append(linear_sampling_audit(session))
    assert audits[-1]['max_linear_resampling_relative_RMSE']<1e-6
    motion=np.loadtxt(RESULTS/session/'func_motion.tsv')
    sc=np.load(RESULTS/session/'sc_fraction.npy')[np.ix_(common,common)]
    fixed_y=np.load(RESULTS/session/'roi_timeseries_raw.npy')
    _,fixed_base,keep,_,_=clean_timeseries(fixed_y,motion,np.empty((len(motion),0)))
    g0=fixed_base[keep].mean(1);g0ss=g0@g0
    images={};masks={};scores={};variance={};pves={};anat={}
    for variant in ['uncorrected','corrected']:
        src=variant_folder(session,variant);out=OUT/session/variant;out.mkdir(exist_ok=True)
        image=nib.load(src/('func_motion_corrected.nii.gz' if variant=='uncorrected' else 'func_sdc.nii.gz'))
        images[variant]=image.get_fdata(dtype=np.float32)
        masks[variant]=legacy_mask(session,variant)
        save_array(masks[variant].astype('uint8'),image,out/'original_noncortical_mask.nii.gz')
        scores[variant],variance[variant]=pca_signals(images[variant],masks[variant])
        np.save(out/'original_pcs20.npy',scores[variant]);np.save(out/'original_variance20.npy',variance[variant])
        # Orthogonal sign/rotation invariance rather than matching arbitrary PC signs.
        match=subspace_similarity(scores[variant][:,:5],np.load(src/'nuisance_pcs.npy'),keep).min()
        assert match>1-1e-8,('Historical PCA reconstruction failed',session,variant,match)
        anat[variant],pves[variant]=anatomical_masks(session,variant,.95)
        for name,mask in [('Noncortical',masks[variant]),*anat[variant].items()]:
            valid_fraction=sum(pves[variant].values())
            maskrows.append({'session':session,'BOLD_processing':variant,'mask':name,'voxels':int(mask.sum()),
                'mean_GM_fraction':float(pves[variant]['GM'][mask].mean()),'mean_WM_fraction':float(pves[variant]['WM'][mask].mean()),
                'mean_CSF_fraction':float(pves[variant]['CSF'][mask].mean()),'unclassified_fraction':float((valid_fraction[mask]<.5).mean())})
        # Cutoff curves use the SAME corrected cortical signals, isolating PC sets.
        for k in range(21):
            f,res,_,q,_=clean_timeseries(fixed_y,motion,scores[variant][:,:k])
            gr=res[keep].mean(1)
            curves.append({'session':session,'PC_source':variant,'components':k,'mean_fc_r':float(f[np.ix_(common,common)][ii,jj].mean()),
                'sc_fc_spearman':sc_fc_coupling(sc,f[np.ix_(common,common)]),'cortical_mean_partial_R2':float(1-gr@gr/g0ss),
                'residual_df':q['residual_df'],'PC_variance_fraction':float(variance[variant][k-1]) if k else np.nan})
        # Complete alternative denoising pipelines; ROI signals stay fixed WITHIN each BOLD variant.
        y=np.load(src/'roi_timeseries_raw.npy')
        combined=anat[variant]['WM']|anat[variant]['CSF']
        raw,_=pca_signals(images[variant],combined)
        filtered,var=pca_signals(images[variant],combined,keep,True)
        wm,_=pca_signals(images[variant],anat[variant]['WM'],keep,True)
        csf,_=pca_signals(images[variant],anat[variant]['CSF'],keep,True)
        means=np.column_stack([images[variant][anat[variant][t]].mean(0) for t in ['WM','CSF']])
        np.save(out/'anatomical_pcs20.npy',filtered);np.save(out/'anatomical_raw_pcs20.npy',raw)
        np.save(out/'anatomical_variance20.npy',var);np.save(out/'wm_csf_means.npy',means)
        modelsets={'motion_dct':np.empty((len(y),0)),'original5':scores[variant][:,:5],
                   'anatomical5_raw':raw[:,:5],'anatomical5':filtered[:,:5],
                   'wm5_csf5':np.column_stack([wm[:,:5],csf[:,:5]]),'wm_csf_means':means}
        _,base,_,_,_=clean_timeseries(y,motion,np.empty((len(y),0)));gsbase=base[keep].mean(1)
        for name,pcs in modelsets.items():
            np.save(out/f'{name}_regressors.npy',pcs)
            f,res,k,q,fd=clean_timeseries(y,motion,pcs)
            assert np.array_equal(k,keep)
            f=f[np.ix_(common,common)];np.save(out/f'fc_{name}.npy',f)
            gs=res[k].mean(1)
            rows.append({'session':session,'BOLD_processing':variant,'model':name,'common_nodes':int(common.sum()),
                'nuisance_regressors':pcs.shape[1],'mean_fc_r':float(f[ii,jj].mean()),'sc_fc_spearman':sc_fc_coupling(sc,f),
                'cortical_mean_partial_R2':float(1-gs@gs/(gsbase@gsbase)),
                'mean_ROI_partial_R2':float(np.mean(1-np.sum(res[k]**2,axis=0)/np.sum(base[k]**2,axis=0))),
                **q,**{key:val for key,val in metrics(np.maximum(f,0),.2,1.).items() if key not in ['nodes']}})
            net,names=network_means(f,groups[common],fisher=True)
            for i,label in enumerate(names):networkrows.append({'session':session,'BOLD_processing':variant,'model':name,'network':label,'within_fc_fisher_z':float(net[i,i])})
        # Prespecified tissue thresholds / PC counts / stricter censoring sensitivity.
        for threshold in [.90,.95,.99]:
            am,_=anatomical_masks(session,variant,threshold)
            pcs,_=pca_signals(images[variant],am['WM']|am['CSF'],keep,True)
            for count in [2,5,10]:
                f,_,_,q,_=clean_timeseries(y,motion,pcs[:,:count]);f=f[np.ix_(common,common)]
                thresholdrows.append({'session':session,'BOLD_processing':variant,'PVE_threshold':threshold,'components':count,'FD_threshold':.5,
                    'WM_voxels':int(am['WM'].sum()),'CSF_voxels':int(am['CSF'].sum()),'mean_fc_r':float(f[ii,jj].mean()),
                    'sc_fc_spearman':sc_fc_coupling(sc,f),'residual_df':q['residual_df']})
        _,_,strictkeep,_,_=clean_timeseries(y,motion,np.empty((len(y),0)),fd_threshold=.3)
        strictpcs,_=pca_signals(images[variant],combined,strictkeep,True)
        f,_,_,q,_=clean_timeseries(y,motion,strictpcs[:,:5],fd_threshold=.3);f=f[np.ix_(common,common)]
        thresholdrows.append({'session':session,'BOLD_processing':variant,'PVE_threshold':.95,'components':5,'FD_threshold':.3,
            'WM_voxels':int(anat[variant]['WM'].sum()),'CSF_voxels':int(anat[variant]['CSF'].sum()),'mean_fc_r':float(f[ii,jj].mean()),
            'sc_fc_spearman':sc_fc_coupling(sc,f),'residual_df':q['residual_df']})
    # Factorial data-versus-mask change, with a matched cubic zero-field control.
    images['cubic_zero_field']=cubic_motion_control(session)
    valid=nib.load(RESULTS/session/'bold_valid_mask.nii.gz').get_fdata()>.5
    diagnostic_masks={**masks,'old_field_valid':masks['uncorrected']&valid,'intersection':masks['uncorrected']&masks['corrected']}
    for dv in ['uncorrected','cubic_zero_field','corrected']:
        for mv,mask in diagnostic_masks.items():
            pcs,_=pca_signals(images[dv],mask)
            f,res,k,q,_=clean_timeseries(fixed_y,motion,pcs[:,:5]);gr=res[k].mean(1)
            factorial.append({'session':session,'image_for_PCA':dv,'mask_for_PCA':mv,'mean_fc_r':float(f[np.ix_(common,common)][ii,jj].mean()),
                'cortical_mean_partial_R2':float(1-gr@gr/g0ss),'mask_voxels':int(mask.sum())})
    old,new=masks['uncorrected'],masks['corrected']
    # Partial component correlations remove identical base motion/DCT terms first.
    a=base_residuals(scores['uncorrected'],motion)[keep]
    b=base_residuals(scores['corrected'],motion)[keep]
    cross=np.corrcoef(a.T,b.T)[:20,20:];np.save(OUT/session/'old_new_PC_partial_correlation.npy',cross)
    for v,x in [('uncorrected',a),('corrected',b)]:
        for j in range(20):
            corr=float(np.corrcoef(x[:,j],g0)[0,1])
            components.append({'session':session,'PC_source':v,'component':j+1,'variance_fraction':float(variance[v][j]),
                'partial_cortical_mean_r':corr,'partial_cortical_mean_r_squared':corr**2})
    sim=subspace_similarity(a,b,np.ones(len(a),bool))
    sim5=subspace_similarity(a[:,:5],b[:,:5],np.ones(len(a),bool))
    q={'session':session,'original_mask_voxels':int(old.sum()),'refit_mask_voxels':int(new.sum()),
       'mask_Dice':float(2*(old&new).sum()/(old.sum()+new.sum())),
       'old_only_voxels':int((old&~new).sum()),'new_only_voxels':int((new&~old).sum()),
       'min_squared_canonical_correlation_5PC_after_base':float(sim5.min()),
       'min_squared_canonical_correlation_20PC_after_base':float(sim.min()),
       'refit_PC5_old_PC6_partial_abs_r':float(abs(cross[5,4])),
       'old_mask_outside_corrected_field_support':int((old&~(nib.load(RESULTS/session/'bold_valid_mask.nii.gz').get_fdata()>.5)).sum()),
       'primary_FC_sha256':sha(RESULTS/session/'fc_primary.npy'),
       'unchanged_SC_sha256':sha(RESULTS/session/'sc_fraction.npy')}
    diagnostics.append(q);print(q,flush=True)

for name,data in [('nuisance_denoising_comparison',rows),('nuisance_mask_composition',maskrows),('nuisance_pc_cutoff',curves),
                  ('nuisance_mask_data_factorial',factorial),('nuisance_component_diagnostics',components),
                  ('nuisance_network_summary',networkrows),('nuisance_anatomical_sensitivity',thresholdrows),('nuisance_sampling_audit',audits)]:
    pd.DataFrame(data).to_csv(RESULTS/f'{name}.csv',index=False)
summary={'method_version':METHOD,'sessions':diagnostics,'PCA_recipes':{'original':'all 180 frames, mean/std normalize, no temporal preprocessing before PCA',
          'anatomical_raw':'same PCA recipe as original; anatomical mask only',
          'anatomical':'constant/linear/<0.01-Hz DCT projection, scaling and PCA fit only on retained original indices'},
         'common_nodes':int(common.sum()),'primary_pipeline':'anatomical5; retained-frame combined WM/CSF five PCs',
         'cautions':['In-sample variance removal is not physiological validation','No physiological recordings or manually validated tissue masks','Crossed image/mask PCA is a diagnostic on the common voxel grid; spatial correspondence differs after unwarping']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))

# Publication figures.
plt.rcParams.update({'font.size':12})
df=pd.DataFrame(curves);comp=pd.DataFrame(components)
fig,axs=plt.subplots(2,3,figsize=(10,6.2))
for row,s in enumerate(SESSIONS):
    for v,c,label in [('uncorrected','#999999','Previous PCs'),('corrected','#328c91','Refitted PCs')]:
        d=df[(df.session==s)&(df.PC_source==v)]
        axs[row,0].plot(d.components,d.mean_fc_r,'o-',ms=3,color=c,label=label)
        axs[row,1].plot(d.components,d.cortical_mean_partial_R2,'o-',ms=3,color=c,label=label)
        d=comp[(comp.session==s)&(comp.PC_source==v)&(comp.component<=10)]
        axs[row,2].plot(d.component,d.partial_cortical_mean_r_squared,'o-',ms=3,color=c,label=label)
    for ax in axs[row]:ax.axvline(5,color='#b45b55',ls=':',lw=1);ax.set_xlabel('PC count / component rank');ax.spines[['top','right']].set_visible(False)
    axs[row,0].set_ylabel(f'{s}\nMean FC r');axs[row,1].set_ylabel('Cortical-mean partial R²');axs[row,2].set_ylabel('Single-PC partial r²')
axs[0,0].legend(fontsize=10);fig.suptitle('PC cutoff sensitivity: corrected cortical signals, fixed motion and censoring',fontsize=12)
fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(RESULTS/'figures/nuisance_pc_mechanism.png',dpi=180);plt.close(fig)
df=pd.DataFrame(rows);names=['motion_dct','original5','anatomical5_raw','anatomical5','wm5_csf5','wm_csf_means']
ticks=['Base','Original 5','Anat. 5 raw','PRIMARY: anat. 5','WM5 + CSF5','Tissue means']
fig,axs=plt.subplots(1,3,figsize=(10,3.8))
for s,c in [('preop','#357da8'),('postop','#d88b3c')]:
    d=df[(df.session==s)&(df.BOLD_processing=='corrected')].set_index('model').loc[names]
    for ax,col in zip(axs,['mean_fc_r','sc_fc_spearman','cortical_mean_partial_R2']):ax.plot(range(len(names)),d[col],'o-',label=s,color=c)
for ax,title in zip(axs,['Mean FC r','SC–FC Spearman correlation','Cortical-mean partial R²']):
    ax.axvspan(2.75,3.25,color='#328c91',alpha=.12)
    ax.set_xticks(range(len(names)),ticks,rotation=45,ha='right',fontsize=11);ax.set_title(title,fontsize=11);ax.spines[['top','right']].set_visible(False)
axs[0].legend(fontsize=10);fig.suptitle('Anatomical WM/CSF denoising: same corrected ROI signals and censoring',fontsize=12)
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(RESULTS/'figures/nuisance_anatomical_comparison.png',dpi=180);plt.close(fig)
from pat16.nuisance_views import show_masks
from PIL import Image
panels=[]
for session in SESSIONS:
    fig=show_masks(session,145)
    fig.savefig(OUT/session/'mask_qc.png',dpi=180);plt.close(fig)
    panels.append(Image.open(OUT/session/'mask_qc.png').convert('RGB'))
combined=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),'white')
offset=0
for panel in panels:combined.paste(panel,(0,offset));offset+=panel.height
combined.save(RESULTS/'figures/nuisance_mask_qc.png')
print(pd.DataFrame(rows).query("BOLD_processing=='corrected'")[["session","model","mean_fc_r","sc_fc_spearman","cortical_mean_partial_R2","residual_df"]].to_string(index=False))
