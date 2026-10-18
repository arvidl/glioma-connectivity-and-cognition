"""Tables and numeric macros are generated, never hand-transcribed."""
import json,shutil
from pathlib import Path
import numpy as np,pandas as pd
from pat16 import ROOT,RESULTS
out=ROOT/'report/generated';out.mkdir(exist_ok=True)
figdir=ROOT/'report/figures';figdir.mkdir(exist_ok=True)
for fig in (RESULTS/'figures').glob('*.png'):shutil.copy2(fig,figdir/fig.name)
s=json.loads((RESULTS/'summary.json').read_text());clinical=pd.read_csv(RESULTS/'assessments_all.csv');net=pd.read_csv(RESULTS/'network_summary.csv');sens=pd.read_csv(RESULTS/'sensitivity.csv');metrics=pd.read_csv(RESULTS/'graph_metrics.csv')

def texsafe(x):
    return str(x).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('—','---')

def fieldlabel(x):
    return r'{\ttfamily\footnotesize '+texsafe(x).replace(r'\_',r'\_\allowbreak{}')+'}'

def number(x):
    if pd.isna(x):return '--'
    try:
        f=float(x)
        return f'{f:.0f}' if abs(f-round(f))<1e-8 else f'{f:.3f}'.rstrip('0').rstrip('.')
    except (ValueError,TypeError):return texsafe(x)

def table(filename,head,rows,fmt):
    text='\\begin{tabular}{'+fmt+'}\n\\toprule\n'+' & '.join(head)+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'
    (out/filename).write_text(text)

qc=[]
for label,fn in [
    ('Seed-160 length-filtered streamlines',lambda x:x['sc_qc']['length_filtered_streamlines']),
    ('Seed-160 assigned interparcel streamlines',lambda x:x['sc_qc']['assigned_interparcel_streamlines']),
    ('Primary retained BOLD frames',lambda x:x['fc_qc']['analyses'][0]['retained_frames']),
    ('Primary retained minutes',lambda x:x['fc_qc']['analyses'][0]['retained_minutes']),
    ('Residual regression degrees of freedom',lambda x:x['fc_qc']['analyses'][0]['residual_df']),
    ('Mean approximate FD (mm)',lambda x:x['fc_qc']['analyses'][0]['mean_fd_mm']),
    ('Own-consensus SC edges',lambda x:x['sc_support']['consensus_edges']),
    ('Common-consensus coupling edges',lambda x:x['sc_support']['common_edges']),
    ('SC--FC Spearman, common consensus',lambda x:x['sc_fc_spearman'])]:
    qc.append([label]+[number(fn(s['sessions'][session])) for session in ['preop','postop']])
table('qc.tex',['Quantity','Pre-op','Post-op'],qc,'lrr')
dwi=[]
for label,key in [('AP/PA b0 correlation before','b0_AP_PA_correlation_before'),('AP/PA b0 correlation after','b0_AP_PA_correlation_after'),('Mean absolute movement RMS (mm)','mean_absolute_movement_rms_mm'),('Maximum absolute movement RMS (mm)','max_absolute_movement_rms_mm'),('Flagged/replaced slice entries','outlier_slices'),('AP nonpositive-Jacobian brain voxels','AP_nonpositive_jacobian_brain_voxels'),('PA nonpositive-Jacobian brain voxels','PA_nonpositive_jacobian_brain_voxels')]:
    dwi.append([label]+[number(s['sessions'][session]['dwi_correction'][key]) for session in ['preop','postop']])
table('dwi_qc.tex',['Quantity','Pre-op','Post-op'],dwi,'lrr')
bqc=pd.read_csv(RESULTS/'bold_correction_qc.csv').set_index('session')
rows=[]
for label,key,scale in [('BOLD readout time (s)','BOLD_readout_seconds',1),('Alternative displacement scale','field_scale_alternative',1),('Reference brain with valid support (\\%)','valid_reference_brain_fraction',100),('Signed shift: 1st percentile (vox)','shift_p01_vox',1),('Signed shift: 99th percentile (vox)','shift_p99_vox',1),('Nonpositive Jacobians in reference brain','nonpositive_jacobian_reference_brain',1)]:
    rows.append([label]+[(f'{bqc.loc[t,key]:.5f}' if key=='BOLD_readout_seconds' else number(bqc.loc[t,key]*scale)) for t in ['preop','postop']])
table('bold_qc.tex',['Quantity','Pre-op','Post-op'],rows,'lrr')
bc=pd.read_csv(RESULTS/'primary_sc/bold_comparison.csv')
blabel={'uncorrected':'Rigid only, matched denoising','corrected':'Static field transfer','readout_alternative':'Alternative timing'}
rows=[]
for _,r in bc.iterrows():
    rows.append([r.session,blabel[r.BOLD_processing],f'{r.mean_fc_r:.3f}',f'{r.sc_fc_spearman:.3f}',f'{r.weighted_global_efficiency:.4f}'])
table('bold_comparison.tex',['Session','BOLD processing','Mean FC $r$',r'SC--FC $\rho$','FC weighted $E$'],rows,'llrrr')
bn=pd.read_csv(RESULTS/'bold_nuisance_sensitivity.csv')
rows=[]
for rv,pv in [('uncorrected','uncorrected'),('uncorrected','corrected'),('corrected','uncorrected'),('corrected','corrected')]:
    d=bn[(bn.ROI_processing==rv)&(bn.PC_processing==pv)].set_index('session')
    rows.append([{'uncorrected':'Previous','corrected':'Corrected'}[rv],{'uncorrected':'Previous','corrected':'Refitted'}[pv]]+[f'{d.loc[t,"mean_fc_r"]:.3f}' for t in ['preop','postop']])
table('bold_nuisance.tex',['ROI signals','Nuisance PCs','Mean FC pre $r$','Mean FC post $r$'],rows,'llrr')
comparison=pd.read_csv(RESULTS/'correction_comparison.csv')
rows=[]
for _,r in comparison.iterrows():
    rows.append([r.session,'Topup/eddy' if r.preprocessing=='corrected' else 'Rigid only',number(r.assigned_streamlines),f'{r.binary_global_efficiency:.4f}',f'{r.weighted_global_efficiency:.5f}',f'{r.sc_fc_spearman:.3f}'])
table('correction_comparison.tex',['Session','Processing','Assigned','Binary $E$','Weighted $E$',r'SC--FC $\rho$'],rows,'llrrrr')
rows=[]
for mod in ['SC','FC_positive']:
    x=metrics[(metrics.modality==mod)&(metrics.target_density==.2)].set_index('session')
    for name in ['edges','components','binary_global_efficiency','weighted_global_efficiency','weighted_clustering']:
        rows.append([texsafe(mod.replace('_',' ')),texsafe(name.replace('_',' ')),number(x.loc['preop',name]),number(x.loc['postop',name])])
table('metrics.tex',['Graph','Metric','Pre-op','Post-op'],rows,'llrr')
rows=[]
for name in dict.fromkeys(net.network):
    d=net[net.network==name].set_index('session')
    rows.append([name]+[number(d.loc[t,'within_sc_weight']) for t in ['preop','postop']]+[number(d.loc[t,'within_fc_fisher_z']) for t in ['preop','postop']])
table('networks.tex',['Network','SC pre / 10k seeds','SC post / 10k seeds','FC pre $z$','FC post $z$'],rows,'lrrrr')
rows=[]
for name in ['primary','fd03','fd10','gsr','bandpass']:
    d=sens[sens.analysis==name].set_index('session')
    vals=[f'{d.loc[t,"mean_fc_r"]:.4f}' if t in d.index else '--' for t in ['preop','postop']]
    rows.append([{'primary':'FD 0.5, high-pass only','fd03':'FD 0.3, high-pass only','fd10':'FD 1.0, high-pass only','gsr':'FD 0.5, whole-brain GSR','bandpass':'FD 0.5, 0.01--0.1 Hz'}[name]]+vals)
table('sensitivity.tex',['Analysis','Mean FC pre $r$','Mean FC post $r$'],rows,'lrr')
selected=['mot_latency_mean','mot_error_mean','rti_simple_rt_mean','rti_five_rt','rvp_a','rvp_latency_mean','soc_prob_min_moves','soc_mean_moves_4','soc_mean_moves_5','ssp_span_length']
names=['MOT mean latency (ms)','MOT mean spatial error (pixels)','RTI simple reaction time (ms)','RTI five-choice reaction time (ms)',"RVP detection A-prime",'RVP mean latency (ms)','SOC minimum-move problems','SOC mean moves: 4-move','SOC mean moves: 5-move','SSP span length']
c=clinical.set_index('measure');rows=[]
for key,name in zip(selected,names):
    r=c.loc[key];rows.append([name,number(r.preop),number(r.postop),number(r.delta_post_minus_pre)])
table('cantab_selected.tex',['Measure','Pre-op','Post-op',r'$\Delta$'],rows,'lrrr')
for name,df in [('cantab_all',clinical[clinical.group=='CANTAB']),('psychosocial_all',clinical[clinical.group!='CANTAB'])]:
    text=r'\begin{longtable}{p{.47\linewidth}rrr}'+'\n'+r'\toprule Measure & Pre-op & Post-op & $\Delta$ \\'+'\n'+r'\midrule\endhead'+'\n'
    for _,r in df.iterrows():text+=f"{fieldlabel(r.measure)} & {number(r.preop)} & {number(r.postop)} & {number(r.delta_post_minus_pre)}"+r' \\'+'\n'
    text+=r'\bottomrule\end{longtable}'+'\n';(out/f'{name}.tex').write_text(text)
    columns=r'p{.22\linewidth}p{.70\linewidth}' if name=='psychosocial_all' else r'p{.26\linewidth}p{.66\linewidth}'
    text=r'\begin{longtable}{'+columns+'}\n'+r'\toprule Source field & Definition, units, direction and interpretation \\'+'\n'+r'\midrule\endhead'+'\n'
    for _,r in df.iterrows():
        detail=texsafe(r.definition)+r'\par\textit{Unit:} '+texsafe(r.units)+r'. \textit{Direction:} '+texsafe(r.direction_text)+r'\par\textit{Limit:} '+texsafe(r.caveat)
        gap='1pt' if name=='psychosocial_all' else '7pt'
        text+=fieldlabel(r.measure)+' & '+detail+r' \\['+gap+']\n'
    text+=r'\bottomrule\end{longtable}'+'\n';(out/f'{name}_dictionary.tex').write_text(text)

den=pd.read_csv(RESULTS/'primary_sc/denoising_comparison.csv')
modelnames={'motion_dct':'Motion/DCT only','original5':'Original noncortical 5 PCs','anatomical5_raw':'Anatomical 5 PCs, raw fit','anatomical5':r'\textbf{Primary: anatomical 5 PCs}','wm5_csf5':'WM5 + CSF5, retained fit','wm_csf_means':'WM and CSF means'}
rows=[]
for model,label in modelnames.items():
    d=den[(den.BOLD_processing=='corrected')&(den.model==model)].set_index('session')
    rows.append([label]+[f'{d.loc[t,"mean_fc_r"]:.3f}' for t in ['preop','postop']]+[f'{d.loc[t,"sc_fc_spearman"]:.3f}' for t in ['preop','postop']]+[str(int(d.loc[t,'residual_df'])) for t in ['preop','postop']])
table('nuisance_models.tex',['Model','FC pre','FC post',r'$\rho$ pre',r'$\rho$ post','df pre','df post'],rows,'lrrrrrr')
mask=pd.read_csv(RESULTS/'nuisance_mask_composition.csv');rows=[]
for t in ['preop','postop']:
    for name in ['Noncortical','WM','CSF']:
        r=mask[(mask.session==t)&(mask.BOLD_processing=='corrected')&(mask['mask']==name)].iloc[0]
        rows.append([t,name,str(r.voxels)]+[f'{r[k]:.3f}' for k in ['mean_WM_fraction','mean_CSF_fraction','mean_GM_fraction']])
table('nuisance_masks.tex',['Session','Mask','EPI voxels','WM fraction','CSF fraction','GM fraction'],rows,'llrrrr')
factors=pd.read_csv(RESULTS/'nuisance_mask_data_factorial.csv');rows=[]
for t in ['preop','postop']:
    for data,label in [('uncorrected','Linear, zero shift'),('cubic_zero_field','Cubic, zero shift'),('corrected','Cubic, static field')]:
        d=factors[(factors.session==t)&(factors.image_for_PCA==data)].set_index('mask_for_PCA')
        rows.append([t,label]+[f'{d.loc[m,"mean_fc_r"]:.3f}' for m in ['uncorrected','corrected','old_field_valid','intersection']])
table('nuisance_factorial.tex',['Session','PCA image','Old mask','New mask','Old valid','Intersection'],rows,'llrrrr')
cut=pd.read_csv(RESULTS/'nuisance_pc_cutoff.csv');rows=[]
for k in [4,5,6,10]:
    d=cut[(cut.session=='postop')&(cut.components==k)].set_index('PC_source')
    rows.append([str(k)]+[f'{d.loc[t,"mean_fc_r"]:.3f}' for t in ['uncorrected','corrected']]+[f'{d.loc[t,"cortical_mean_partial_R2"]:.3f}' for t in ['uncorrected','corrected']])
table('nuisance_cutoff.tex',['PCs','Old mean FC','Refit mean FC',r'Old partial $R^2$',r'Refit partial $R^2$'],rows,'rrrrr')
adj=pd.read_csv(RESULTS/'primary_sc/pc_sensitivity.csv');rows=[]
for count in [2,4,5,6,10]:
    d=adj[(adj.BOLD_processing=='corrected')&(adj.components==count)].set_index('session')
    rows.append([str(count)]+[f'{d.loc[t,"mean_fc_r"]:.4f}' for t in ['preop','postop']]+[f'{d.loc[t,"sc_fc_spearman"]:.3f}' for t in ['preop','postop']])
table('primary_pc_sensitivity.tex',['PCs','FC pre','FC post',r'$\rho$ pre',r'$\rho$ post'],rows,'rrrrr')
cd=pd.read_csv(RESULTS/'censoring_investigation/decomposition.csv')
rows=[]
stage_labels={'primary':'Primary: own frames/fit','restriction':'Strict frames: fixed primary fit',
              'coefficients':'+ refit regression, fixed PCs','pca':'+ refit tissue PCA (strict pipeline)'}
for stage,label in stage_labels.items():
    d=cd[cd.stage==stage].set_index('session')
    rows.append([label,f'{d.loc["preop","coupling"]:.4f}',f'{d.loc["postop","coupling"]:.4f}',
                 f'{d.loc["postop","coupling"]-d.loc["preop","coupling"]:+.4f}'])
table('censoring_decomposition.tex',['Ordered diagnostic',r'$\rho$ pre',r'$\rho$ post',r'$\Delta\rho$'],rows,'lrrr')
cm=pd.read_csv(RESULTS/'censoring_investigation/matched_summary.csv');rows=[]
match_labels={'duration':'Duration','motion':'Duration + FD','motion_time':'Duration + FD + half'}
for _,r in cm.iterrows():
    rows.append([r.pool.title(),match_labels[r.matching],{'frozen':'Frozen','refit':'Refitted'}[r['mode']],
                 f'{r.coupling_delta_median:+.4f}',f'[{r.coupling_delta_q025:+.4f}, {r.coupling_delta_q975:+.4f}]',
                 f'{100*r.coupling_positive_fraction:.1f}'])
table('censoring_matching.tex',['Pool','Matching','Fit',r'Median $\Delta\rho$','Central 95\\% spread',r'$\Delta>0$ (\%)'],rows,'lllrrr')
rows=[]
for stage,label in [('primary','Primary'),('pca','Strict FD')]:
    d=cd[cd.stage==stage].set_index('session')
    for field,support in [('coupling','Own SC support/weights'),('common_coupling','Common support, own weights'),('fixed_pre_sc_coupling','Common support, fixed pre SC')]:
        rows.append([label,support,f'{d.loc["preop",field]:.4f}',f'{d.loc["postop",field]:.4f}',f'{d.loc["postop",field]-d.loc["preop",field]:+.4f}'])
table('censoring_support.tex',['Frames/fit','SC reference',r'$\rho$ pre',r'$\rho$ post',r'$\Delta\rho$'],rows,'llrrr')
rows=[]
for _,r in cm[cm['mode']=='refit'].iterrows():
    rows.append([r.pool.title(),match_labels[r.matching],f'{r.fd_abs_smd_median:.3f}',
                 f'{r.fd_abs_smd_max_abs:.3f}',f'{r.fd_wasserstein_mm_median:.4f}',
                 f'{r.first_half_count_difference_median:+.0f}'])
table('censoring_balance.tex',['Pool','Matching','Med. $|$SMD$|$','Max. $|$SMD$|$','Med. $W_1$ (mm)',r'Med. $\Delta n_{half1}$'],rows,'llrrrr')
def edge_driver_tables():
    p=RESULTS/'coupling_edge_drivers'
    groups=pd.read_csv(p/'support_summary.csv').set_index('support_group')
    glabel={'shared':'Shared SC','pre_only':'Pre-op-only SC','post_only':'Post-op-only SC'}
    rows=[]
    for name in ['shared','pre_only','post_only']:
        r=groups.loc[name]
        rows.append([glabel[name],str(int(r.edges)),f'{r.primary_delta_contribution:+.4f}',
                     f'{r.pca_delta_contribution:+.4f}',f'{r.reversal_shift:+.4f}',
                     number(r.preop_median_count)+' / '+number(r.postop_median_count)])
    table('edge_support.tex',['SC support','Edges',r'$\Delta$ primary',r'$\Delta$ strict',r'$R$ term','Median counts pre/post'],rows,'lrrrrr')
    net=pd.read_csv(p/'network_pairs.csv')
    chosen=pd.concat([net.nlargest(5,'reversal_shift'),net.nsmallest(3,'reversal_shift')])
    rows=[]
    for _,r in chosen.iterrows():
        rows.append([r['pair'].replace(' / ','--'),str(int(r.edges))]+[f'{r[k]:+.4f}' for k in ['shared_shift','pre_only_shift','post_only_shift','reversal_shift']])
    table('edge_networks.tex',['Network pair','Edges','Shared','Pre-only','Post-only',r'Total $R$ term'],rows,'lrrrrr')
    e=pd.read_csv(p/'edges.csv');chosen=pd.concat([e.nlargest(4,'reversal_shift'),e.nsmallest(4,'reversal_shift')])
    rows=[]
    for _,r in chosen.iterrows():
        label=''
        for q in ['i','j']:
            v=f'{int(r["node_"+q])}: '+r['label_'+q].replace('7Networks_','')
            label+=r'{\ttfamily '+texsafe(v).replace(r'\_',r'\_\allowbreak{}')+'}'+(r'\newline ' if q=='i' else '')
        rows.append([label,{'shared':'Shared','pre_only':'Pre-only','post_only':'Post-only'}[r.support_group],
                     f'{r.sc_pre:.0f} / {r.sc_post:.0f}',f'{1000*r.reversal_shift:+.3f}'])
    table('edge_individuals.tex',['Node IDs and Schaefer100 atlas labels','SC support','Counts pre/post',r'$10^3 r_e$'],rows,r'p{.46\linewidth}lrr')
    d=pd.read_csv(p/'deletions.csv');rows=[]
    chosen=d[d.kind.isin(['baseline','support','network_incident']) | ((d.kind=='top_edges')&(d.removed=='positive_10'))]
    labels={'none':'None (original)','shared':'All shared SC','pre_only':'All pre-op-only SC','post_only':'All post-op-only SC',
            'all_session_specific':'Both session-specific SC groups','positive_10':'Top 10 positive edge terms'}
    for _,r in chosen.iterrows():
        label=labels.get(r.removed,r.removed+' incident edges')
        rows.append([label,str(int(r.union_edges_removed)),f'{r.primary_delta:+.4f}',f'{r.pca_delta:+.4f}',f'{r.reversal_shift:+.4f}'])
    table('edge_deletions.tex',['Edges removed','Union edges',r'$\Delta$ primary',r'$\Delta$ strict',r'$R$ after removal'],rows,'lrrrr')
    m=pd.read_csv(p/'matched_group_summary.csv');m=m[(m['mode']=='refit')&(m.partition=='support')];rows=[]
    for protocol,label in [('strict_duration','Duration'),('strict_motion','Duration + FD'),('strict_motion_time','Duration + FD + half')]:
        v=m[m.protocol==protocol].set_index('group').mean_contribution
        rows.append([label]+[f'{v[g]:+.4f}' for g in ['shared','pre_only','post_only']]+[f'{v.sum():+.4f}'])
    table('edge_matched_support.tex',['Strict-pool matching','Shared term','Pre-only term','Post-only term',r'Mean $\Delta$'],rows,'lrrrr')
edge_driver_tables()
def sc_reliability_tables():
    p=RESULTS/'sc_reliability';e=pd.read_csv(p/'edges.csv')
    names={'shared':'Shared','pre_only':'Pre-op-only','post_only':'Post-op-only','absent_both':'Absent in both'}
    rows=[]
    for group in names:
        v=e[e.baseline_group==group];both=(v.repeat_consensus80_class=='both').sum()
        rows.append([names[group],str(len(v)),f'{v.preop_frequency_ge1.median():.2f}',f'{v.postop_frequency_ge1.median():.2f}',
                     str(int((v.preop_frequency_ge1>0).sum())),str(int((v.postop_frequency_ge1>0).sum())),str(int(both))])
    table('sc_repeat_support.tex',['Original support','Edges','Med. pre $p$','Med. post $p$','Ever pre','Ever post',r'Both $p\geq.8$'],rows,'lrrrrrr')
    rows=[]
    for lo,hi,label in [(1,1,'1'),(2,2,'2'),(3,4,'3--4'),(5,9,'5--9'),(10,999999,r'$\geq10$')]:
        vals=[]
        for session,base in [('preop','pre'),('postop','post')]:
            v=e[e[f'baseline_{base}_count'].between(lo,hi)]
            vals.append((len(v),v[session+'_frequency_ge1'].median(),v[session+'_repeat_cv'].median()))
        rows.append([label,f'{vals[0][0]} / {vals[1][0]}',f'{vals[0][1]:.2f} / {vals[1][1]:.2f}',f'{vals[0][2]:.2f} / {vals[1][2]:.2f}'])
    table('sc_repeat_counts.tex',['Original count','Edges pre/post','Median detection pre/post','Median CV pre/post'],rows,'lrrr')
    cross=pd.read_csv(p/'original_reversal_by_reliability.csv');cross=cross[cross.baseline_group!='absent_both'];rows=[]
    for group,label in [('both','Both'),('pre_only','Pre-op only'),('post_only','Post-op only'),('neither','Neither')]:
        v=cross[cross.repeat_consensus80_class==group].set_index('baseline_group')
        vals=[float(v.loc[g,'original_reversal_term']) for g in ['shared','pre_only','post_only']]
        rows.append([label,str(int(v.edges.sum()))]+[f'{x:+.4f}' for x in vals]+[f'{sum(vals):+.4f}'])
    table('sc_repeat_reversal_classes.tex',[r'80\% repeat class','Union edges','Original shared','Original pre-only','Original post-only','Total $R$ term'],rows,'lrrrrr')
    f=pd.read_csv(p/'fixed_coupling.csv');r=pd.read_csv(p/'repeat_coupling_summary.csv');rows=[]
    for row in f[(f.method=='baseline')&(f.support=='own')].itertuples():
        v=r[(r.threshold==row.threshold)&(r.support=='own')].iloc[0]
        rows.append([str(row.threshold),f'{row.pre_edges} / {row.post_edges}',f'{row.primary_delta:+.4f}',f'{row.strict_delta:+.4f}',
            f'{v.primary_delta_median:+.4f}',f'{v.strict_delta_median:+.4f}',f'{v.reversal_pattern_fraction:.3f}'])
    table('sc_repeat_coupling.tex',['Min. count','Edges pre/post',r'Original $\Delta_P$',r'Original $\Delta_S$',r'Repeat med. $\Delta_P$',r'Repeat med. $\Delta_S$','Flip fraction'],rows,'lrrrrrr')
    rows=[]
    for row in f[(f.method=='consensus')&(f.threshold==1)].itertuples():
        rows.append([row.support,f'{row.consensus_fraction:.2f}',f'{row.pre_edges} / {row.post_edges}',f'{row.primary_delta:+.4f}',f'{row.strict_delta:+.4f}',f'{row.reversal_shift:+.4f}'])
    table('sc_repeat_consensus.tex',['SC support','Min. occurrence','Edges pre/post',r'$\Delta_P$',r'$\Delta_S$',r'$R$'],rows,'llrrrr')
    m=pd.read_csv(p/'matched_coupling_summary.csv');rows=[]
    names={'baseline_k1':r'Original $k\geq1$','baseline_k3':r'Original $k\geq3$','baseline_k5':r'Original $k\geq5$','consensus80_own':r'80\% own consensus','consensus80_common':r'80\% common consensus'}
    for ref,label in names.items():
        vals=[]
        for protocol in ['strict_duration','strict_motion','strict_motion_time']:
            v=m[(m.protocol==protocol)&(m.sc_reference==ref)].iloc[0]
            vals.append(f'{v["median"]:+.3f} [{v.q025:+.3f}, {v.q975:+.3f}]')
        rows.append([label]+vals)
    table('sc_repeat_matched.tex',['Fixed SC reference','Duration','Duration + FD','Duration + FD + half'],rows,'lrrr')
sc_reliability_tables()
primary_sc=pd.read_csv(RESULTS/'primary_sc/coupling_sensitivity.csv')
rows=[]
for design,support,label in [('original_k1','own','Historical single run, own'),('repeat_mean_unfiltered','own','Unfiltered repeat mean, own'),('consensus80_k1','own','80\\% consensus, own'),('consensus80_k1','common',r'\textbf{Primary coupling: common}')]:
    r=primary_sc[(primary_sc.design==design)&(primary_sc.support==support)].iloc[0]
    rows.append([label,str(int(r.pre_edges)),str(int(r.post_edges)),f'{r.primary_pre:.4f}',f'{r.primary_post:.4f}',f'{r.primary_delta:+.4f}',f'{r.strict_delta:+.4f}'])
table('primary_sc_choice.tex',['SC population','Pre $n$','Post $n$',r'$\rho_P$ pre',r'$\rho_P$ post',r'$\Delta_P$',r'$\Delta_S$'],rows,'lrrrrrr')
sg=pd.read_csv(RESULTS/'primary_sc/graph_sensitivity.csv')
rows=[]
for design,support,label in [('original_fraction','own','Single run, assigned-total fraction'),('original_k1','own','Single run, per-seed rate'),('repeat_mean_unfiltered','own','Repeat mean, unfiltered'),('consensus80_k1','own',r'\textbf{Primary graphs: own consensus}'),('consensus80_k1','common','Common consensus sensitivity')]:
    d=sg[(sg.design==design)&(sg.support==support)&(sg.target_density==.2)].set_index('session')
    vals=[d.loc[t,'weighted_global_efficiency'] for t in ['preop','postop']]
    rows.append([label,str(int(d.loc['preop','edges'])),str(int(d.loc['postop','edges'])),f'{vals[0]:.5f}',f'{vals[1]:.5f}',f'{100*(vals[1]/vals[0]-1):+.2f}'])
table('primary_sc_graph_sensitivity.tex',['SC construction','Pre $n$','Post $n$','$E$ pre','$E$ post',r'$\Delta E$ (\%)'],rows,'lrrrrr')
dec=pd.read_csv(RESULTS/'primary_sc/censoring_decomposition.csv')
table('primary_sc_decomposition.tex',['Ordered FC fit',r'$\rho$ pre',r'$\rho$ post',r'$\Delta$'],[[stage_labels[r.stage],f'{r.pre_rho:.4f}',f'{r.post_rho:.4f}',f'{r.delta:+.4f}'] for _,r in dec.iterrows()],'lrrr')
matched=pd.read_csv(RESULTS/'primary_sc/matched_coupling_summary.csv')
table('primary_sc_matched.tex',['Pool','Matching',r'Median $\Delta$',r'Conditional 95\% spread'],[[r.protocol.split('_',1)[0].title(),match_labels[r.protocol.split('_',1)[1]],f'{r["median"]:+.4f}',f'[{r.q025:+.4f}, {r.q975:+.4f}]'] for _,r in matched.iterrows()],'llrr')
joint=pd.read_csv(RESULTS/'joint_sampling/summary.csv')
jprotocols=['primary_block10','strict_block10','primary_duration','primary_motion','primary_motion_time','strict_duration','strict_motion','strict_motion_time']
jlabels=['Full primary, block 10','Full strict, block 10','Primary: duration','Primary: duration + FD','Primary: duration + FD + half','Strict: duration','Strict: duration + FD','Strict: duration + FD + half']
for mode in ['fixed','reselected']:
    rows=[]
    for protocol,label in zip(jprotocols,jlabels):
        r=joint[(joint.protocol==protocol)&(joint.support==mode)&(joint.source=='joint')].iloc[0]
        rows.append([label,f'{r["median"]:+.4f}',f'[{r.q025:+.4f}, {r.q975:+.4f}]',f'{100*r.positive_fraction:.2f}',f'{100*r.SC_share:.1f}',f'{100*r.FC_share:.1f}'])
    table('joint_'+mode+'.tex',['FC design',r'Median $\Delta$','Conditional 95\\% range',r'$\Delta>0$ (\%)','SC (\\%)','FC (\\%)'],rows,'lrrrrr')
rows=[]
for protocol,pool in [('primary_block10','Primary'),('strict_block10','Strict')]:
    for source in ['SC_only','FC_only','joint']:
        r=joint[(joint.protocol==protocol)&(joint.support=='fixed')&(joint.source==source)].iloc[0]
        rows.append([pool,texsafe(source.replace('_',' ')),f'{r["median"]:+.4f}',f'[{r.q025:+.4f}, {r.q975:+.4f}]',f'{r.sd:.4f}'])
table('joint_sources.tex',['Full-run FC','Perturbation',r'Median $\Delta$','Conditional 95\\% range','SD'],rows,'llrrr')
session_joint=pd.read_csv(RESULTS/'joint_sampling/session_coupling.csv')
rows=[]
for protocol,pool in [('primary_block10','Primary'),('strict_block10','Strict')]:
    for mode in ['fixed','reselected']:
        v=session_joint[(session_joint.protocol==protocol)&(session_joint.support==mode)&(session_joint.source=='joint')].set_index('session')
        cells=[f'{v.loc[t,"median"]:.4f} [{v.loc[t,"q025"]:.4f}, {v.loc[t,"q975"]:.4f}]' for t in ['preop','postop']]
        rows.append([pool,mode]+cells)
table('joint_sessions.tex',['Full-run FC','Support',r'Pre-op $\rho$: median [range]',r'Post-op $\rho$: median [range]'],rows,'llrr')
rows=[]
for pool in ['primary','strict']:
    for block in [5,10,20]:
        for mode in ['fixed','reselected']:
            r=joint[(joint.protocol==f'{pool}_block{block}')&(joint.support==mode)&(joint.source=='joint')].iloc[0]
            rows.append([pool.title(),str(block),mode,f'{r["median"]:+.4f}',f'[{r.q025:+.4f}, {r.q975:+.4f}]'])
table('joint_blocks.tex',['Censor pool','Block frames','Support',r'Median $\Delta$','Conditional 95\\% range'],rows,'lrlrr')

refits=pd.read_csv(RESULTS/'joint_refit/summary.csv')
mode_labels={'frozen':'Frozen','coefficients':'Refit coefficients','full_refit':'Refit PCs + coefficients'}
rows=[]
for support in ['fixed','reselected']:
    for mode in mode_labels:
        r=refits[(refits.protocol=='primary_block10')&(refits.support==support)&(refits['mode']==mode)].iloc[0]
        rows.append([support,mode_labels[mode],f'{r["median"]:+.4f}',f'[{r.q025:+.4f}, {r.q975:+.4f}]',f'{100*r.positive_fraction:.1f}',f'{100*r.FC_share:.1f}'])
table('joint_refit.tex',['Support','Nuisance training',r'Median $\Delta$','Conditional 95\\% range',r'$\Delta>0$ (\%)','FC (\\%)'],rows,'llrrrr')
feasibility=pd.read_csv(RESULTS/'joint_refit/feasibility.csv');rows=[]
for r in feasibility.itertuples():
    rows.append([r.protocol.split('_')[0].title(),r.protocol.split('block')[1],f'{r.preop_eligible}/200',f'{r.postop_eligible}/200',f'{r.paired_eligible}/200',f'{r.preop_unique_median:g}/{r.postop_unique_median:g}'])
table('joint_refit_feasibility.tex',['Pool','Block','Pre eligible','Post eligible','Paired eligible','Median distinct pre/post'],rows,'llrrrr')
shifts=pd.read_csv(RESULTS/'joint_refit/paired_shifts.csv');rows=[]
shift_labels={'coefficients_minus_frozen':'Coefficients minus frozen','full_minus_coefficients':'Full minus coefficients','full_minus_frozen':'Full minus frozen'}
for r in shifts[(shifts.protocol=='primary_block10')&(shifts.support=='fixed')].itertuples():
    rows.append([shift_labels[r.contrast],f'{r.median:+.4f}',f'[{r.q025:+.4f}, {r.q975:+.4f}]'])
table('joint_refit_shifts.tex',['Paired contrast shift','Median','Conditional 95\\% range'],rows,'lrr')
rows=[]
for block in [5,10,20]:
    for support in ['fixed','reselected']:
        for mode in mode_labels:
            r=refits[(refits.protocol==f'primary_block{block}')&(refits.support==support)&(refits['mode']==mode)].iloc[0]
            rows.append([str(block),support,mode_labels[mode],f'{r["median"]:+.4f}',f'[{r.q025:+.4f}, {r.q975:+.4f}]'])
table('joint_refit_blocks.tex',['Block','Support','Nuisance training',r'Median $\Delta$','Conditional 95\\% range'],rows,'rllrr')
evidence=pd.read_csv(RESULTS/'evidence_integration/network_uncertainty.csv');rows=[]
for name in dict.fromkeys(evidence.network):
    v=evidence[evidence.network==name]
    sc=v[v.modality=='SC'].iloc[0];frozen=v[v['mode']=='frozen'].iloc[0];full=v[v['mode']=='full_refit'].iloc[0]
    rows.append([name,f'{sc.observed_delta:+.3f}',f'[{sc.q025:+.3f}, {sc.q975:+.3f}]',f'{full.observed_delta:+.3f}',f'[{frozen.q025:+.3f}, {frozen.q975:+.3f}]',f'[{full.q025:+.3f}, {full.q975:+.3f}]'])
table('evidence_networks.tex',['Network',r'SC $\Delta$','SC range',r'FC $\Delta z$','FC frozen range','FC refit range'],rows,'lrrrrr')
rows=[]
d=lambda key:float(c.loc[key,'delta_post_minus_pre'])
for task,profile,link in [
 ('MOT',f'Latency {d("mot_latency_mean"):+.1f} ms; spatial error {d("mot_error_mean"):+.3f} pixels','SomMot; Vis'),
 ('RTI',f'Simple RT {d("rti_simple_rt_mean"):+.2f} ms; correct counts unchanged','DorsAttn; SalVentAttn; SomMot'),
 ('RVP',f'A-prime {d("rvp_a"):+.5f}; false alarms {number(c.loc["rvp_false_alarms","preop"])} to {number(c.loc["rvp_false_alarms","postop"])}','DorsAttn; SalVentAttn; Cont'),
 ('SOC',f'Minimum-move problems {number(c.loc["soc_prob_min_moves","preop"])}/12; 4/5-move changes differ','Cont; DorsAttn'),
 ('SSP',f'Span {number(c.loc["ssp_span_length","preop"])} to {number(c.loc["ssp_span_length","postop"])}','DorsAttn; Cont; Vis'),
 ('Context',f'STAI {d("stai"):+.0f}; PSWQ {d("pswq"):+.0f}; loneliness {d("loneliness"):+.0f}; sick leave','No network assigned')]:
    rows.append([task,texsafe(profile),texsafe(link)])
table('evidence_tasks.tex',['Evidence','Observed profile','Candidate networks (hypotheses)'],rows,r'lp{.45\linewidth}p{.30\linewidth}')

# Sampling mechanisms: every displayed number comes from saved experimental rows.
pcdir=RESULTS/'pc_sampling'
pc=pd.read_csv(pcdir/'summary.csv')
pc_names={'bootstrap':'Bootstrap','unit_coverage':'Same frames, unit weights','block_unique':'Blocks, matched count','block80':'Blocks, 80\\% retained'}
mechanism_names={'coefficients':'Frozen PCs; refit regression','temporal_only':'Temporal projection only','preprocessing_only':'Preprocessing only','pca_only':'PCA loadings only','full_refit':'Preprocessing + loadings'}
def pc_interval(r,digits=4):return f'[{r.q025:+.{digits}f}, {r.q975:+.{digits}f}]'
rows=[]
for design,label in pc_names.items():
    for mode in mode_labels:
        r=pc[(pc.protocol==design+'_block10')&(pc.support=='fixed')&(pc['mode']==mode)].iloc[0]
        rows.append([label,mode_labels[mode],f'{r["median"]:+.4f}',pc_interval(r)])
table('pc_sampling_summary.tex',['Sampling','Nuisance fit',r'Median $\Delta$','Conditional range'],rows,'llrr')
pqc=pd.read_csv(pcdir/'sampling_qc.csv');rows=[]
for design,label in pc_names.items():
    for session in ['preop','postop']:
        q=pqc[(pqc.design==design)&(pqc.block==10)&(pqc.session==session)]
        rows.append([label,session,f'{q.rows.median():g}',f'{q.unique.median():g}',f'{q.frequency_Kish_count.median():.1f}',f'{q.mean_fd.median():.3f}'])
table('pc_sampling_qc.tex',['Sampling','Session','Rows','Distinct','Kish count','Mean FD'],rows,'llrrrr')
mech=pd.read_csv(pcdir/'mechanism_summary.csv');rows=[]
for r in mech.itertuples():rows.append([mechanism_names[r.mode],f'{r.median:+.5f}',pc_interval(r,5)])
table('pc_sampling_mechanism.tex',['Training intervention',r'Median $\Delta$','Conditional range'],rows,'lrr')
rows=[]
ms=pd.read_csv(pcdir/'mechanism_shifts.csv')
short={'temporal_minus_coefficients':'Temporal only minus frozen PCs','preprocessing_minus_coefficients':'Preprocessing only minus frozen PCs','full_minus_preprocessing':'Full minus preprocessing only','pca_minus_coefficients':'Loadings only minus frozen PCs','full_minus_pca':'Full minus loadings only','full_minus_coefficients':'Full minus frozen PCs'}
for r in ms.itertuples():rows.append([short[r.contrast],f'{r.median:+.5f}',pc_interval(r,5)])
table('pc_sampling_shifts.tex',['Paired shift in longitudinal contrast','Median','Conditional range'],rows,'lrr')
pm=pd.read_csv(pcdir/'fc_metrics.csv');rows=[]
for design,label in pc_names.items():
    for session in ['preop','postop']:
        q=pm[(pm.protocol==design+'_block10')&(pm.session==session)&(pm['mode']=='full_refit')]
        rows.append([label,session,f'{q.adjusted_PC_cos2_min.median():.3f}',f'{q.gap5_relative.median():.3f}',f'{100*q.PCA5_fraction.median():.1f}',f'{100*q.cortical_mean_PC_removed.median():.1f}',f'{q.mean_fc_r.median():.3f}'])
table('pc_sampling_stability.tex',['Sampling','Session',r'Min $\cos^2$',r'Gap$_{5/6}$',r'PC var. (\%)',r'Mean removed (\%)','FC $r$'],rows,'llrrrrr')
rows=[]
for block in [5,10,20]:
    for design,label in pc_names.items():
        r=pc[(pc.protocol==f'{design}_block{block}')&(pc.support=='fixed')&(pc['mode']=='full_refit')].iloc[0]
        rows.append([str(block),label,f'{r["median"]:+.4f}',pc_interval(r)])
table('pc_sampling_blocks.tex',['Block','Sampling',r'Median $\Delta$','Conditional range'],rows,'rlrr')
pn=pd.read_csv(pcdir/'network_ranges.csv');rows=[];range_rows=[]
for network in dict.fromkeys(pn.network):
    v=pn[(pn.network==network)&(pn['mode']=='full_refit')].set_index('protocol').loc[[d+'_block10' for d in pc_names]]
    rows.append([network,f'{v.observed_delta.iloc[0]:+.4f}']+[f'{x:+.4f}' for x in v['median']])
    range_rows.append([network]+[pc_interval(r,3) for r in v.itertuples()])
table('pc_sampling_network_centers.tex',['Network','Observed', 'Bootstrap','Unit weights','Blocks, matched',r'Blocks, 80\%'],rows,'lrrrrr')
table('pc_sampling_network_ranges.tex',['Network','Bootstrap','Unit weights','Blocks, matched',r'Blocks, 80\%'],range_rows,'lrrrr')

macros={'LesionVolume':number(s['lesion_volume_ml']),'CommonNodes':str(s['common_fc_nodes']),'CantabCount':str(sum(clinical.group=='CANTAB')),'AssessmentCount':str(len(clinical))}
for protocol,tag in [('primary_block10','JointPrimary'),('strict_block10','JointStrict'),('strict_motion','JointStrictMotion'),('strict_motion_time','JointStrictHalf')]:
    for mode,suffix in [('fixed',''),('reselected','Reselect')]:
        r=joint[(joint.protocol==protocol)&(joint.support==mode)&(joint.source=='joint')].iloc[0]
        for col,end in [('median','Median'),('q025','Low'),('q975','High')]:
            macros[tag+suffix+end]=f'{r[col]:+.5f}'
        for col,end in [('SC_share','SCShare'),('FC_share','FCShare'),('interaction_share','InteractionShare'),('positive_fraction','Positive')]:
            macros[tag+suffix+end]=f'{100*r[col]:.2f}'
q=pd.read_csv(RESULTS/'joint_sampling/support_qc.csv')
halves=pd.read_csv(RESULTS/'joint_sampling/monte_carlo_halves.csv')
tail=halves[(halves.protocol=='strict_motion')&(halves.support=='reselected')&halves['half'].str.startswith('FC_')].q975
macros['JointTailHalfMin']=f'{tail.min():+.5f}'
macros['JointTailHalfMax']=f'{tail.max():+.5f}'
for col,tag in [('common_edges','JointSupport'),('jaccard','JointJaccard')]:
    for prob,end in [(.025,'Low'),(.5,'Median'),(.975,'High')]:
        macros[tag+end]=f'{q[col].quantile(prob):.1f}' if col=='common_edges' else f'{q[col].quantile(prob):.4f}'
for session,prefix in [('preop','Pre'),('postop','Post')]:
    for processing,tag in [('corrected',''),('uncorrected','Old')]:
        r=den[(den.session==session)&(den.BOLD_processing==processing)&(den.model=='anatomical5')].iloc[0]
        macros[prefix+'AnatFC'+tag]=f'{r.mean_fc_r:.3f}'
        macros[prefix+'AnatCoupling'+tag]=f'{r.sc_fc_spearman:.3f}'
    r=den[(den.session==session)&(den.BOLD_processing=='corrected')&(den.model=='original5')].iloc[0]
    macros[prefix+'LegacyFC']=f'{r.mean_fc_r:.3f}'
    macros[prefix+'LegacyCoupling']=f'{r.sc_fc_spearman:.3f}'
    x=s['sessions'][session];a=x['fc_qc']['analyses'][0]
    macros[prefix+'ConsensusEdges']=str(x['sc_support']['consensus_edges'])
    for name,value in [('Streamlines',x['sc_qc']['length_filtered_streamlines']),('Assigned',x['sc_qc']['assigned_interparcel_streamlines']),('Frames',a['retained_frames']),('Minutes',a['retained_minutes']),('Dof',a['residual_df']),('Coupling',x['sc_fc_spearman'])]:macros[prefix+name]=number(value)
for session,prefix in [('preop','Pre'),('postop','Post')]:
    m=metrics[(metrics.session==session)&(metrics.target_density==.2)].set_index('modality')
    macros[prefix+'SCComponents']=str(int(m.loc['SC','components']))
    macros[prefix+'SCBinaryE']=f"{m.loc['SC','binary_global_efficiency']:.4f}"
    macros[prefix+'SCWeightedE']=f"{m.loc['SC','weighted_global_efficiency']:.5f}"
    macros[prefix+'FCWeightedE']=f"{m.loc['FC_positive','weighted_global_efficiency']:.4f}"
    for analysis,label in [('primary','Primary'),('fd03','Strict'),('gsr','GSR')]:
        value=sens[(sens.session==session)&(sens.analysis==analysis)].iloc[0].mean_fc_r
        macros[prefix+'MeanFC'+label]=f"{value:.4f}" if analysis=='gsr' else f"{value:.3f}"
    for variant,label in [('uncorrected','Old'),('readout_alternative','Alt')]:
        r=bc[(bc.session==session)&(bc.BOLD_processing==variant)].iloc[0]
        macros[prefix+'MeanFC'+label]=f'{r.mean_fc_r:.3f}'
        macros[prefix+'Coupling'+label]=f'{r.sc_fc_spearman:.3f}'
    for radius,label in [(2,'Two'),(8,'Eight')]:
        value=sens[(sens.session==session)&(sens.analysis==f'SC_radius_{radius}mm')].iloc[0].assigned_streamlines
        macros[prefix+'Assigned'+label]=f"{value:.0f}"
    for network,label in [('SomMot','SomMot'),('Cont','Control')]:
        value=net[(net.session==session)&(net.network==network)].iloc[0].within_fc_fisher_z
        macros[prefix+label+'Z']=f"{value:.3f}"
m=metrics[(metrics.modality=='SC')&(metrics.target_density==.2)].set_index('session')
macros['SCWeightedPercentChange']=f"{100*(m.loc['postop','weighted_global_efficiency']/m.loc['preop','weighted_global_efficiency']-1):+.1f}"
r=primary_sc[(primary_sc.design=='consensus80_k1')&(primary_sc.support=='common')].iloc[0]
macros['PrimarySCDelta']=f'{r.primary_delta:+.5f}'
macros['StrictSCDelta']=f'{r.strict_delta:+.5f}'
macros['PreStrictCoupling']=f'{r.strict_pre:.3f}'
macros['PostStrictCoupling']=f'{r.strict_post:.3f}'
for mode,tag in [('frozen','RefitFrozen'),('coefficients','RefitCoefficients'),('full_refit','RefitFull')]:
    r=refits[(refits.protocol=='primary_block10')&(refits.support=='fixed')&(refits['mode']==mode)].iloc[0]
    for col,end in [('median','Median'),('q025','Low'),('q975','High')]:macros[tag+end]=f'{r[col]:+.5f}'
    macros[tag+'Positive']=f'{100*r.positive_fraction:.2f}'
    macros[tag+'FCShare']=f'{100*r.FC_share:.2f}'
# Recommended FC sampling outputs; preceding sensitivity tables stay unchanged.
pfc=pd.read_csv(RESULTS/'primary_fc/summary.csv')
rows=[]
for protocol,mode,support,label in [
    ('two_thirds_block10','full_refit','fixed','97, block 10, refit (primary)'),
    ('two_thirds_block10','frozen','fixed','97, block 10, frozen'),
    ('two_thirds_block10','coefficients','fixed','97, block 10, coef.'),
    ('two_thirds_block5','full_refit','fixed','97, block 5, refit'),
    ('two_thirds_block20','full_refit','fixed','97, block 20, refit'),
    ('half_block10','full_refit','fixed','73, block 10, refit'),
    ('four_fifths_block10','full_refit','fixed','116, block 10, refit'),
    ('two_thirds_block10','full_refit','reselected','97, block 10, SC reselected')]:
    r=pfc[(pfc.protocol==protocol)&(pfc['mode']==mode)&(pfc.support==support)].iloc[0]
    rows.append([label,f'{r['median']:+.5f}',f'[{r.q025:+.5f}, {r.q975:+.5f}]'])
table('primary_fc_coupling.tex',['Frames / design','Median','Conditional range'],rows,'lrr')
pn=pd.read_csv(RESULTS/'primary_fc/network_uncertainty.csv').query("role=='primary'");rows=[]
for network in pn.network.unique():
    a=pn[(pn.network==network)&(pn.modality=='SC')].iloc[0];b=pn[(pn.network==network)&(pn.modality=='FC')].iloc[0]
    rows.append([texsafe(network),f'{a.observed_delta:+.3f}',f'[{a.q025:+.3f}, {a.q975:+.3f}]',f'{b.observed_delta:+.3f}',f'{b['median']:+.3f}',f'[{b.q025:+.3f}, {b.q975:+.3f}]'])
table('primary_fc_networks.tex',['Network',r'SC $\Delta$', 'SC range', r'FC $\Delta z$','FC median','FC range'],rows,'lrrrrr')
r=pfc.query("role=='primary'").iloc[0]
for col,tag in [('median','Median'),('q025','Low'),('q975','High')]:macros['PrimaryFC'+tag]=f'{r[col]:+.5f}'
for col,tag in [('FC_share','FCShare'),('SC_share','SCShare'),('interaction_share','InteractionShare')]:macros['Primary'+tag]=f'{100*r[col]:.2f}'
(out/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items()))
print('Generated report tables and macros')
