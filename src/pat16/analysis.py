"""Generate every reported result, table and figure from bundled inputs."""
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from . import ROOT,DATA,RESULTS,SESSIONS
from .clinical import assessment_tables,HYPOTHESES
from .connectivity import labels,clean_timeseries
from .primary_sc import load_weights, VERSION as SC_VERSION
from .networks import metrics,sc_fc_coupling,network_means,block_bootstrap_network,attenuate,ou_fc,graph_growth

COLORS=['#7759a4','#3d7da4','#2b9b82','#d4952e','#bb655a','#5f6c9b','#bb709d']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
FIG=RESULTS/'figures'


def finish(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)


def node_table():
    d=labels()
    atlasimg=nib.load(RESULTS/'anatomy/preop_atlas.nii.gz');a=atlasimg.get_fdata().astype(int)
    l=nib.load(RESULTS/'anatomy/preop_lesion.nii.gz').get_fdata()>.5
    distance=ndi.distance_transform_edt(~l,sampling=atlasimg.header.get_zooms()[:3])
    for i in range(1,101):
        mask=a==i;idx=i-1
        vox=np.argwhere(mask);centre=nib.affines.apply_affine(atlasimg.affine,vox.mean(0))
        for key,value in zip(['x','y','z'],centre):d.loc[idx,key]=value
        d.loc[idx,'voxels']=mask.sum();d.loc[idx,'lesion_overlap_fraction']=(mask&l).sum()/max(mask.sum(),1)
        d.loc[idx,'distance_to_lesion_mm']=distance[mask].min() if mask.any() else np.nan
    d.to_csv(RESULTS/'nodes.csv',index=False)
    return d


def anatomy_figure():
    pre=nib.as_closest_canonical(nib.load(DATA/'sub-PAT16/ses-preop/anat/sub-PAT16_ses-preop_T1w.nii.gz'))
    a=pre.get_fdata();b=nib.as_closest_canonical(nib.load(RESULTS/'anatomy/postop_in_preop.nii.gz')).get_fdata()
    l=nib.as_closest_canonical(nib.load(RESULTS/'anatomy/preop_lesion.nii.gz')).get_fdata()>.5
    at=nib.as_closest_canonical(nib.load(RESULTS/'anatomy/preop_atlas.nii.gz')).get_fdata()
    centre=np.argwhere(l).mean(0).astype(int);cuts=np.clip(centre[2]+np.array([-15,0,20]),0,a.shape[2]-1)
    fig,axs=plt.subplots(2,3,figsize=(10,6))
    for row,vol in enumerate([a,b]):
        for j,k in enumerate(cuts):
            ax=axs[row,j];ax.imshow(vol[:,:,k].T,origin='lower',cmap='gray',vmin=0,vmax=np.percentile(vol[vol>0],99))
            if row==0:ax.contour(at[:,:,k].T>0,levels=[.5],colors=['#4ab7b0'],linewidths=.35)
            if l[:,:,k].any():ax.contour(l[:,:,k].T,levels=[.5],colors=['#ffca54'],linewidths=.8)
            ax.set_xlim(8,a.shape[0]-8);ax.set_ylim(45,215);ax.set_xticks([]);ax.set_yticks([])
            ax.set_title(f'{"Pre-op" if row==0 else "Post-op aligned to pre-op"} | z={nib.affines.apply_affine(pre.affine,[0,0,k])[2]:.0f} mm',fontsize=9)
    fig.suptitle('Anatomy, cortical atlas and preoperative lesion footprint',fontsize=13)
    fig.text(.5,.015,'Neurological orientation (patient right on image right). Yellow is the pre-op lesion in both rows, not a cavity segmentation.',ha='center',fontsize=8)
    fig.subplots_adjust(top=.9,bottom=.07,wspace=.03,hspace=.12);finish(fig,'anatomy')


def registration_figure():
    fig,axs=plt.subplots(2,3,figsize=(10,6.5))
    for row,s in enumerate(SESSIONS):
        t=nib.as_closest_canonical(nib.load(DATA/f'sub-PAT16/ses-{s}/anat/sub-PAT16_ses-{s}_T1w.nii.gz'));a=t.get_fdata()
        for col,kind in enumerate(['dwi','func','atlas']):
            if kind=='atlas':b=nib.as_closest_canonical(nib.load(RESULTS/f'anatomy/{s}_atlas.nii.gz')).get_fdata()
            else:b=nib.as_closest_canonical(nib.load(RESULTS/f'{s}/{kind}_in_t1.nii.gz')).get_fdata()
            k=int(a.shape[2]*.57);ax=axs[row,col];ax.imshow(a[:,:,k].T,origin='lower',cmap='gray',vmax=np.percentile(a[a>0],99))
            if kind=='atlas':ax.imshow(np.ma.masked_equal(b[:,:,k].T,0),origin='lower',cmap='turbo',alpha=.45,vmin=1,vmax=100)
            else:
                levels=np.percentile(b[b>0],[35,60,80]);ax.contour(b[:,:,k].T,levels=levels,colors=['#50c5bc','#ffc857','#e16565'],linewidths=.6)
            ax.set_title(f'{s}: {kind} on T1',fontsize=10);ax.set_xticks([]);ax.set_yticks([]);ax.set_ylim(45,220)
    fig.suptitle('Registration QC: EPI contours and atlas labels in session T1 space',fontsize=12)
    fig.text(.5,.015,'Topup/eddy DWI; static topup-field correction of BOLD. Rigid EPI registration and affine atlas projection remain approximate.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.96]);finish(fig,'registration_qc')


def run():
    FIG.mkdir(exist_ok=True,parents=True)
    clinical=assessment_tables();nodes=node_table();groups=nodes.network.to_numpy()
    sc={s:load_weights(s,'own') for s in SESSIONS}
    coupling_sc={s:load_weights(s,'common') for s in SESSIONS}
    fc={s:np.load(RESULTS/f'{s}/fc_primary.npy') for s in SESSIONS}
    valid={s:np.isfinite(np.load(RESULTS/f'{s}/roi_timeseries_raw.npy')).all(0) for s in SESSIONS}
    common=valid['preop']&valid['postop'];np.save(RESULTS/'common_fc_nodes.npy',common)
    for s in SESSIONS:
        fc[s][~common,:]=np.nan;fc[s][:,~common]=np.nan
    summary={'subject':'sub-PAT16','analysis_date':'2026-10-18','primary_sc_policy':SC_VERSION,'primary_sc_decision':'Post hoc methodological recommendation; config/PRIMARY_SC.md','primary_denoising_model':'anatomical5','primary_denoising_decision':'Post hoc methodological recommendation; config/PRIMARY_DENOISING.md','evidence_level':'exploratory paired single case; no inference of surgery effects','common_fc_nodes':int(common.sum()),'sessions':{},'limitations':['BOLD static cross-acquisition field assumption; uncertain readout and shim compatibility','slice timing uncorrected','FC depends on the recommended anatomical nuisance model; segmentation and neural specificity lack independent validation','affine template and rigid longitudinal registration','no normative cognition or retest data','no direct QoL scale','no surgical/treatment interval or medication details in bundled tables']}
    rows=[];sensitivity=[];network_rows=[]
    scscale=max(sc['preop'].max(),sc['postop'].max())
    for s in SESSIONS:
        q=json.loads((RESULTS/f'{s}/fc_qc.json').read_text())
        qs=json.loads((RESULTS/f'{s}/sc_qc.json').read_text())
        summary['sessions'][s]={'sc_qc':qs,'fc_qc':q,'dwi_correction':json.loads((RESULTS/s/'dwi_correction.json').read_text()),'bold_correction':json.loads((RESULTS/s/'bold_sdc.json').read_text()),'sc_support':json.loads((RESULTS/'primary_sc/policy.json').read_text())['sessions'][s],'sc_fc_support':'common 16/20 consensus','sc_weight_units':'assigned streamlines per 10000 attempted seeds','sc_fc_spearman':sc_fc_coupling(coupling_sc[s][np.ix_(common,common)],fc[s][np.ix_(common,common)])}
        for density in [.1,.15,.2,.25,.3]:
            for modality,w,scale in [('SC',sc[s],scscale),('FC_positive',np.maximum(fc[s][np.ix_(common,common)],0),1.)]:
                row={'session':s,'modality':modality,'target_density':density,**metrics(w,density,scale)};rows.append(row)
        for name in ['primary','fd03','fd10','gsr','bandpass']:
            path=RESULTS/f'{s}/fc_{name}.npy'
            if not path.exists():continue
            f=np.load(path)[np.ix_(common,common)];i,j=np.triu_indices(len(f),1)
            sensitivity.append({'session':s,'analysis':name,'mean_fc_r':float(np.nanmean(f[i,j])),'sc_fc_spearman':sc_fc_coupling(coupling_sc[s][np.ix_(common,common)],f),'SC_reference':SC_VERSION})
        for radius in [2,4,6,8]:
            w=np.load(RESULTS/f'{s}'/('sc_counts.npy' if radius==6 else f'sc_radius_{radius}mm.npy'))
            sensitivity.append({'session':s,'analysis':f'SC_radius_{radius}mm','assigned_streamlines':int(w.sum()/2),'edges':int(np.count_nonzero(np.triu(w,1)))})
        net_sc,names=network_means(sc[s],groups)
        net_fc,_=network_means(fc[s],groups,fisher=True)
        ts=np.load(RESULTS/f'{s}/roi_timeseries_clean.npy');ts[:,~common]=np.nan
        ci=block_bootstrap_network(ts,np.load(RESULTS/f'{s}/fc_primary_keep.npy'),groups)
        for i,name in enumerate(names):
            network_rows.append({'session':s,'network':name,'within_sc_weight':net_sc[i,i],'SC_reference':SC_VERSION,'within_fc_fisher_z':net_fc[i,i],'fc_z_bootstrap_low':ci[0,i],'fc_z_bootstrap_high':ci[2,i]})
    metrics_table=pd.DataFrame(rows);metrics_table.to_csv(RESULTS/'graph_metrics.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(RESULTS/'sensitivity.csv',index=False)
    nets=pd.DataFrame(network_rows);nets.to_csv(RESULTS/'network_summary.csv',index=False)
    li=nib.load(RESULTS/'anatomy/preop_lesion.nii.gz');summary['lesion_volume_ml']=float((li.get_fdata()>.5).sum()*abs(np.linalg.det(li.affine[:3,:3]))/1000)
    summary['reported_tumor_volume_ml']=50.24
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2))
    anatomy_figure();registration_figure()
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for row,mod in enumerate(['SC','FC']):
        a,b=(sc['preop'],sc['postop']) if mod=='SC' else (fc['preop'],fc['postop'])
        mats=[np.log1p(a),np.log1p(b),b-a] if mod=='SC' else [a,b,b-a]
        for j,m in enumerate(mats):
            kwargs={'cmap':'magma','vmin':0,'vmax':np.log1p(max(sc['preop'].max(),sc['postop'].max()))} if mod=='SC' and j<2 else {'cmap':'RdBu_r','vmin':-1,'vmax':1}
            if mod=='SC' and j==2:kwargs.update(vmin=-np.max(np.abs(b-a)),vmax=np.max(np.abs(b-a)))
            im=axs[row,j].imshow(m,**kwargs);axs[row,j].set_title(f'{mod}: '+['pre-op','post-op','post minus pre'][j]);axs[row,j].set_xlabel('Atlas parcel');axs[row,j].set_ylabel('Atlas parcel');fig.colorbar(im,ax=axs[row,j],shrink=.75)
    fig.suptitle('Patient-derived connectomes | SC log(1 + mean count / 10,000 seeds), FC Pearson r',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.96]);finish(fig,'connectomes')
    fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True)
    names=list(dict.fromkeys(groups));x=np.arange(len(names))
    for i,s in enumerate(SESSIONS):
        d=nets[nets.session==s].set_index('network').loc[names]
        axs[0].bar(x+(i-.5)*.32,d.within_sc_weight,width=.32,label=s,color=['#357da8','#d88b3c'][i])
        yy=d.within_fc_fisher_z.to_numpy();lo=d.fc_z_bootstrap_low.to_numpy();hi=d.fc_z_bootstrap_high.to_numpy()
        axs[1].errorbar(x+(i-.5)*.15,yy,yerr=np.vstack([np.maximum(yy-lo,0),np.maximum(hi-yy,0)]),fmt='o',capsize=3,label=s,color=['#357da8','#d88b3c'][i])
    axs[0].set_ylabel('Mean SC per 10,000 seeds');axs[1].set_ylabel('Mean FC Fisher z');axs[1].set_xticks(x,names);axs[0].legend(ncol=2);axs[1].axhline(0,color='.7',lw=.6)
    fig.suptitle('Within-network summaries; FC intervals are within-run block resampling',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.95]);finish(fig,'networks')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for s,c in zip(SESSIONS,['#357da8','#d88b3c']):
        for ax,mod in zip(axs,['SC','FC_positive']):
            d=metrics_table[(metrics_table.session==s)&(metrics_table.modality==mod)]
            ax.plot(d.target_density,d.weighted_global_efficiency,'o-',label=s,color=c);ax.set_title(mod);ax.set_xlabel('Requested proportional edge density');ax.set_ylabel('Weighted global efficiency');ax.legend()
    fig.tight_layout();finish(fig,'threshold_sensitivity')
    selected=['mot_latency_mean','mot_error_mean','rti_simple_rt_mean','rti_five_rt','rvp_a','rvp_latency_mean','soc_prob_min_moves','soc_mean_moves_4','soc_mean_moves_5','ssp_span_length']
    d=clinical.set_index('measure').loc[selected]
    lab=['MOT mean latency','MOT mean error','RTI simple RT','RTI five-choice RT',"RVP A-prime",'RVP mean latency','SOC minimum-move problems','SOC 4-move mean moves','SOC 5-move mean moves','SSP span']
    fig,ax=plt.subplots(figsize=(10,5));val=d.direction_aligned_percent.to_numpy(float)
    ax.barh(np.arange(len(d)),val,color=np.where(val>=0,'#2f8c80','#bf6262'))
    for i,(_,r) in enumerate(d.iterrows()):
        ax.text(max(val.max(),35)+3,i,f'{float(r.preop):.3g} to {float(r.postop):.3g}',va='center',fontsize=9)
    ax.set_yticks(np.arange(len(d)),lab);ax.invert_yaxis();ax.axvline(0,color='.4',lw=.8);ax.set_xlim(min(val.min()-5,-30),max(val.max(),35)+40)
    ax.set_xlabel('Direction-aligned raw change (%) | rightward = nominally favorable')
    ax.set_title('CANTAB: mixed raw changes; no normative or reliable-change inference');finish(fig,'cognition')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    keys=['stai','pswq','loneliness','erq_reappraisal','erq_suppression'];d=clinical.set_index('measure').loc[keys]
    y=np.arange(len(keys));pre=d.preop.astype(float);post=d.postop.astype(float)
    axs[0].hlines(y,np.minimum(pre,post),np.maximum(pre,post),color='.65');axs[0].scatter(pre,y,label='preop',color='#357da8');axs[0].scatter(post,y,label='postop',color='#d88b3c');axs[0].set_yticks(y,['Anxiety (STAI)','Worry (PSWQ)','Loneliness','Reappraisal','Suppression']);axs[0].set_xlabel('Raw questionnaire score');axs[0].legend()
    axs[1].axis('off');axs[1].text(.02,.8,'Context broadens the interpretation',fontsize=12,weight='bold');axs[1].text(.02,.65,'Employment: employed to sick leave\nAnxiety decreased, worry and loneliness increased.\nNo dedicated health-related QoL score is available.\nScores are not comparable across questionnaires.\nNo validated global cognition/QoL composite is created.',va='top',linespacing=1.8,fontsize=10)
    fig.tight_layout();finish(fig,'psychosocial')
    # Hypothetical lesions use patient-derived baseline geometry/SC, no fitting to cognition.
    exposure=np.exp(-nodes.distance_to_lesion_mm.to_numpy()/10)
    levels=np.linspace(0,1,21);base=sc['preop'];scale=base.max();rho=max(np.linalg.eigvalsh(base).max(),1e-12)
    baseline_eff=metrics(base,1,scale)['weighted_global_efficiency'];basefc=ou_fc(base,.6,scale=rho)
    sim=[]
    for severity in levels:
        injured=attenuate(base,exposure,severity);ef=metrics(injured,1,scale)['weighted_global_efficiency'];pred=ou_fc(injured,.6,scale=rho)
        sim.append({'severity':severity,'relative_efficiency':ef/baseline_eff,'OU_fc_change_frobenius':float(np.linalg.norm(pred-basefc))})
    sims=pd.DataFrame(sim);sims.to_csv(RESULTS/'simulation_lesion_sweep.csv',index=False)
    growth=graph_growth(base,int(np.argmax(nodes.lesion_overlap_fraction)),steps=100,dt=.2,rho=.3)
    np.save(RESULTS/'simulation_graph_growth.npy',growth)
    fig,axs=plt.subplots(1,3,figsize=(12,3.7));axs[0].plot(sims.severity,sims.relative_efficiency,color='#7759a4');axs[0].set_xlabel('Hypothetical injury severity');axs[0].set_ylabel('Efficiency / baseline');axs[0].set_ylim(0,1.05)
    axs[1].plot(sims.severity,sims.OU_fc_change_frobenius,color='#d4952e');axs[1].set_xlabel('Hypothetical injury severity');axs[1].set_ylabel('OU FC change (Frobenius norm)')
    for name,color in zip(names,COLORS):axs[2].plot(np.arange(len(growth))*.2,growth[:,groups==name].mean(1),label=name,color=color)
    axs[2].set_xlabel('Simulated time (arbitrary units)');axs[2].set_ylabel('Simulated concentration');axs[2].legend(fontsize=6,loc='upper left')
    fig.suptitle('Teaching simulations on reconstructed pre-op SC: no patient growth or cognitive prediction',fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);finish(fig,'simulations')
    print('Generated tables, summary and figures',flush=True)
