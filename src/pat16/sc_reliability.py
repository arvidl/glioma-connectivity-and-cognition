"""Conditional seed-repeat stability and count-threshold SC–FC diagnostics."""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr
from . import ROOT,RESULTS,SESSIONS
from .bold import sha
from .tract_repeats import OUT,REPEATS,SEEDS
from .edge_drivers import rank_contributions

THRESHOLDS=[1,2,3,5,10,20]
CONSENSUS=[.5,.8,.95,1.]
VERSION='sc-reliability-analysis-v1'


def load():
    i,j=np.triu_indices(100,1)
    baseline=np.array([np.load(RESULTS/s/'sc_counts.npy')[i,j] for s in SESSIONS])
    repeats=np.array([np.load(OUT/s/'counts.npy')[:,i,j] for s in SESSIONS])
    fc={stage:np.array([np.load(RESULTS/s/f'fc_{tag}.npy')[i,j] for s in SESSIONS]) for stage,tag in [('primary','primary'),('strict','fd03')]}
    return i,j,baseline,repeats,fc


def evaluate(weights,fc):
    """Each support is re-ranked; zeros exclude SC edges rather than impute FC."""
    row={'pre_edges':int((weights[0]>0).sum()),'post_edges':int((weights[1]>0).sum()),
         'shared_edges':int((weights>0).all(0).sum()),'union_edges':int((weights>0).any(0).sum())}
    for stage in ['primary','strict']:
        rho=[rank_contributions(w,f).sum() for w,f in zip(weights,fc[stage])]
        row.update({stage+'_pre':rho[0],stage+'_post':rho[1],stage+'_delta':rho[1]-rho[0]})
    row['reversal_shift']=row['strict_delta']-row['primary_delta']
    return row


def threshold_weights(counts,k,support):
    mask=counts>=k
    if support=='common':mask[:]=mask.all(0)
    return np.where(mask,counts,0.)


def consensus_weights(repeats,k,p,support):
    mask=(repeats>=k).mean(1)>=p
    if support=='common':mask[:]=mask.all(0)
    return np.where(mask,repeats.mean(1),0.)


def edge_inventory(i,j,baseline,repeats):
    nodes=pd.read_csv(RESULTS/'nodes.csv').set_index('id')
    e=pd.DataFrame({'edge_index':np.arange(len(i)),'node_i':i+1,'node_j':j+1,
        'label_i':nodes.loc[i+1,'label'].values,'label_j':nodes.loc[j+1,'label'].values,
        'network_i':nodes.loc[i+1,'network'].values,'network_j':nodes.loc[j+1,'network'].values,
        'baseline_pre_count':baseline[0],'baseline_post_count':baseline[1]})
    e['pair']=[' / '.join(sorted([a,b])) for a,b in zip(e.network_i,e.network_j)]
    e['baseline_group']=np.select([(baseline>0).all(0),(baseline[0]>0)&(baseline[1]==0),(baseline[0]==0)&(baseline[1]>0)],['shared','pre_only','post_only'],default='absent_both')
    for t,s in enumerate(SESSIONS):
        mean=repeats[t].mean(0);sd=repeats[t].std(0,ddof=1)
        e[s+'_repeat_mean_count']=mean;e[s+'_repeat_sd_count']=sd
        e[s+'_repeat_cv']=np.divide(sd,mean,out=np.full(len(i),np.nan),where=mean>0)
        for k in THRESHOLDS:e[f'{s}_frequency_ge{k}']=(repeats[t]>=k).mean(0)
    a=e.preop_frequency_ge1>=.8;b=e.postop_frequency_ge1>=.8
    e['repeat_consensus80_class']=np.select([a&b,a&~b,~a&b],['both','pre_only','post_only'],default='neither')
    e['repeat_pre_detected_post_never']=(e.preop_frequency_ge1>=.8)&(e.postop_frequency_ge1==0)
    e['repeat_post_detected_pre_never']=(e.postop_frequency_ge1>=.8)&(e.preop_frequency_ge1==0)
    original=pd.read_csv(RESULTS/'coupling_edge_drivers/edges.csv').set_index(['node_i','node_j']).reversal_shift
    e['original_reversal_term']=[original.get((a,b),0.) for a,b in zip(e.node_i,e.node_j)]
    e.to_csv(OUT/'edges.csv',index=False)
    e.groupby(['baseline_group','repeat_consensus80_class']).agg(edges=('edge_index','size'),original_reversal_term=('original_reversal_term','sum')).reset_index().to_csv(OUT/'original_reversal_by_reliability.csv',index=False)
    rows=[]
    for t,s in enumerate(SESSIONS):
        bins=np.select([baseline[t]==0,baseline[t]==1,baseline[t]==2,(baseline[t]>=3)&(baseline[t]<=4),(baseline[t]>=5)&(baseline[t]<=9)],['0','1','2','3–4','5–9'],default='10+')
        for group in ['shared','pre_only','post_only','absent_both']:
            for count_bin in ['all','0','1','2','3–4','5–9','10+']:
                mask=(e.baseline_group==group).values
                if count_bin!='all':mask &= bins==count_bin
                if not mask.any():continue
                prob=(repeats[t][:,mask]>=1).mean(0)
                rows.append({'session':s,'baseline_group':group,'baseline_count_bin':count_bin,'edges':int(mask.sum()),
                    'median_detection_fraction':float(np.median(prob)),'mean_detection_fraction':float(prob.mean()),
                    'ever_detected_edges':int((prob>0).sum()),'detected_ge80_edges':int((prob>=.8).sum()),
                    'never_detected_edges':int((prob==0).sum()),'median_mean_count':float(np.median(repeats[t][:,mask].mean(0))),
                    'median_count_cv':float(e.loc[mask,s+'_repeat_cv'].median())})
    pd.DataFrame(rows).to_csv(OUT/'edge_reliability_summary.csv',index=False)
    # Fixed first/last ten split; validation of repeat occurrence, not anatomy.
    rows=[]
    for t,s in enumerate(SESSIONS):
        for k in [1,3,5]:
            for train,test,direction in [(repeats[t,:10],repeats[t,10:],'first_to_last'),(repeats[t,10:],repeats[t,:10],'last_to_first')]:
                a=(train>=k).mean(0)>=.8;b=(test>=k).mean(0)>=.8
                rows.append({'session':s,'threshold':k,'split':direction,'training_consensus_edges':int(a.sum()),
                    'heldout_consensus_edges':int(b.sum()),'consensus_intersection':int((a&b).sum()),
                    'consensus_union':int((a|b).sum()),'consensus_jaccard':float((a&b).sum()/(a|b).sum()),
                    'mean_heldout_detection_in_selected':float((test[:,a]>=k).mean()),
                    'selected_but_never_heldout':int(((test[:,a]>=k).sum(0)==0).sum())})
    pd.DataFrame(rows).to_csv(OUT/'split_half.csv',index=False)
    return e


def repeat_couplings(baseline,repeats,fc):
    fixed=[];contributions=[]
    e=pd.read_csv(OUT/'edges.csv')
    for k in THRESHOLDS:
        for support in ['own','common']:
            for p in [None]+CONSENSUS:
                w=threshold_weights(baseline,k,support) if p is None else consensus_weights(repeats,k,p,support)
                method='baseline' if p is None else 'consensus'
                fixed.append({'method':method,'threshold':k,'consensus_fraction':p,'support':support,**evaluate(w,fc)})
                c={stage:np.array([rank_contributions(a,b) for a,b in zip(w,fc[stage])]) for stage in ['primary','strict']}
                r=(c['strict'][1]-c['strict'][0])-(c['primary'][1]-c['primary'][0])
                for partition,col in [('original_support','baseline_group'),('repeat_consensus80','repeat_consensus80_class'),('network_pair','pair')]:
                    for name,idx in e.groupby(col).groups.items():
                        contributions.append({'method':method,'threshold':k,'consensus_fraction':p,'support':support,'partition':partition,'group':name,
                            'retained_union_edges':int((w[:,idx]>0).any(0).sum()),'reversal_term':float(r[idx].sum())})
    pd.DataFrame(fixed).to_csv(OUT/'fixed_coupling.csv',index=False)
    pd.DataFrame(contributions).to_csv(OUT/'coupling_contributions.csv',index=False)
    runs=[];overlap=[];paired=[]
    for k in THRESHOLDS:
        for t,s in enumerate(SESSIONS):
            for a in range(REPEATS):
                w=np.where(repeats[t,a]>=k,repeats[t,a],0)
                row={'session':s,'repeat':a,'threshold':k,'edges':int((w>0).sum())}
                for stage in fc:row[stage+'_rho']=float(rank_contributions(w,fc[stage][t]).sum())
                runs.append(row)
                for b in range(a+1,REPEATS):
                    x=repeats[t,a]>=k;y=repeats[t,b]>=k
                    overlap.append({'session':s,'threshold':k,'repeat_a':a,'repeat_b':b,'intersection':int((x&y).sum()),
                        'union':int((x|y).sum()),'jaccard':float((x&y).sum()/(x|y).sum()),
                        'count_spearman_all_4950_edges':float(spearmanr(repeats[t,a],repeats[t,b]).statistic)})
        # All Cartesian combinations, not an arbitrary pairing of repeat IDs.
        for a in range(REPEATS):
            for b in range(REPEATS):
                counts=np.stack([repeats[0,a],repeats[1,b]])
                for support in ['own','common']:
                    w=threshold_weights(counts,k,support)
                    paired.append({'pre_repeat':a,'post_repeat':b,'threshold':k,'support':support,**evaluate(w,fc)})
    pd.DataFrame(runs).to_csv(OUT/'run_coupling.csv',index=False)
    pd.DataFrame(overlap).to_csv(OUT/'within_session_overlap.csv',index=False)
    paired=pd.DataFrame(paired);paired.to_csv(OUT/'repeat_pair_coupling.csv',index=False)
    rows=[]
    for (k,support),g in paired.groupby(['threshold','support']):
        row={'threshold':k,'support':support,'combinations':len(g),'reversal_pattern_fraction':float(((g.primary_delta<0)&(g.strict_delta>0)).mean())}
        for key in ['primary_delta','strict_delta','reversal_shift','pre_edges','post_edges','shared_edges']:
            row[key+'_median']=float(g[key].median());row[key+'_q025']=float(g[key].quantile(.025));row[key+'_q975']=float(g[key].quantile(.975))
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'repeat_coupling_summary.csv',index=False)


def matched_controls(baseline,repeats):
    """Hold five SC references fixed while replaying all previous full FC refits."""
    from .censoring import OUT as CENSOR,PROTOCOLS,load_session,fit_selected
    settings={f'baseline_k{k}':threshold_weights(baseline,k,'own') for k in [1,3,5]}
    settings['consensus80_own']=consensus_weights(repeats,1,.8,'own')
    settings['consensus80_common']=consensus_weights(repeats,1,.8,'common')
    masks=np.load(CENSOR/'matched_masks.npz');data=[load_session(s) for s in SESSIONS];i,j=np.triu_indices(100,1)
    reference=pd.read_csv(CENSOR/'matched_replicates.csv');reference=reference[reference['mode']=='refit']
    rows=[]
    for protocol in PROTOCOLS:
        ref=reference[reference.protocol==protocol].pivot(index='replicate',columns='session',values='coupling')
        for rep,ks in enumerate(masks[protocol]):
            fcs=[fit_selected(d,k,'refit')[0][i,j] for d,k in zip(data,ks)]
            for name,w in settings.items():
                rho=[float(rank_contributions(sc,f).sum()) for sc,f in zip(w,fcs)]
                if name=='baseline_k1' and not np.allclose(rho,ref.loc[rep,list(SESSIONS)].values,atol=1e-9):raise RuntimeError('Previous matched FC/SC result not reproduced')
                rows.append({'protocol':protocol,'replicate':rep,'sc_reference':name,'pre_coupling':rho[0],'post_coupling':rho[1],
                    'coupling_delta':rho[1]-rho[0],'pre_edges':int((w[0]>0).sum()),'post_edges':int((w[1]>0).sum())})
        print('SC reliability matched controls:',protocol,flush=True)
    m=pd.DataFrame(rows);m.to_csv(OUT/'matched_coupling.csv',index=False)
    rows=[]
    for (protocol,ref),g in m.groupby(['protocol','sc_reference']):
        v=g.coupling_delta
        rows.append({'protocol':protocol,'sc_reference':ref,'draws':len(v),'median':float(v.median()),'q025':float(v.quantile(.025)),
            'q975':float(v.quantile(.975)),'positive_fraction':float((v>0).mean())})
    pd.DataFrame(rows).to_csv(OUT/'matched_coupling_summary.csv',index=False)


def run():
    sources=[OUT/s/n for s in SESSIONS for n in ['counts.npy','complete.json','baseline_replay.json']]
    sources+=[RESULTS/s/n for s in SESSIONS for n in ['sc_counts.npy','fc_primary.npy','fc_fd03.npy']]
    sources += [RESULTS/'nodes.csv',RESULTS/'coupling_edge_drivers/edges.csv',RESULTS/'censoring_investigation/design.json',RESULTS/'censoring_investigation/matched_masks.npz',RESULTS/'censoring_investigation/matched_replicates.csv']
    spec={'version':VERSION,'code_sha256':sha(ROOT/'src/pat16/sc_reliability.py'),'inputs':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'thresholds_inclusive_counts_per_30000_seeds':THRESHOLDS,'consensus_detection_fractions':CONSENSUS,
        'uncertainty_scope':'conditional random seeding only; fixed acquisitions/CSD/geometry/FC; empirical dependent repeat-combination spreads, not CIs'}
    target=OUT/'analysis_design.json'
    if target.exists() and json.loads(target.read_text())==spec:
        print('SC reliability analysis cache current',flush=True);return
    i,j,b,r,fc=load();edge_inventory(i,j,b,r);repeat_couplings(b,r,fc);matched_controls(b,r)
    (OUT/'summary.json').write_text(json.dumps({'repeats_per_session':REPEATS,'seeds_per_repeat':SEEDS,'independent_random_seed_runs':2*REPEATS,
        'baseline_replays':2,'total_new_repeat_seeds':int(2*REPEATS*SEEDS),'cartesian_combinations':REPEATS**2,
        'thresholds':THRESHOLDS,'consensus_fractions':CONSENSUS,'fixed_primary_results_unchanged':True,
        'interpretation':spec['uncertainty_scope']},indent=2))
    target.write_text(json.dumps(spec,indent=2))


def plot_results():
    import matplotlib.pyplot as plt
    e=pd.read_csv(OUT/'edges.csv');g=pd.read_csv(OUT/'edge_reliability_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for t,s in enumerate(SESSIONS):
        ax=axes[t]
        for name,color in [('shared','#245a81'),('pre_only','#bd6e28'),('post_only','#5d9b87')]:
            x=e.loc[e.baseline_group==name,s+'_frequency_ge1']
            ax.hist(x,bins=np.linspace(-.025,1.025,22),histtype='step',lw=1.6,label=name.replace('_',' '),color=color)
        ax.set_title(s+': original support groups');ax.set_xlabel('Fraction detected in 20 new 30k-seed runs');ax.set_ylabel('Edges');ax.legend(fontsize=8)
    fig.savefig(RESULTS/'figures/sc_repeat_detection.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    overlap=pd.read_csv(OUT/'within_session_overlap.csv')
    for ax,t,s in zip(axes,range(2),SESSIONS):
        x=e[s+'_repeat_mean_count'];p=e[s+'_frequency_ge1'];k=x>0
        ax.scatter(x[k],p[k],s=5,alpha=.25,color='#245a81');ax.set_xscale('log');ax.set_ylim(-.03,1.04)
        ax.set_xlabel('Mean count per 30,000 seeds (20 runs)');ax.set_ylabel('Detection fraction');ax.set_title(s+'; high occurrence does not validate anatomy')
    fig.savefig(RESULTS/'figures/sc_repeat_counts.png',dpi=180);plt.close(fig)
    fixed=pd.read_csv(OUT/'fixed_coupling.csv');summary=pd.read_csv(OUT/'repeat_coupling_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,support in zip(axes,['own','common']):
        a=fixed[(fixed.method=='baseline')&(fixed.support==support)].sort_values('threshold');b=summary[summary.support==support].sort_values('threshold')
        for stage,color in [('primary','#245a81'),('strict','#b35322')]:
            ax.plot(a.threshold,a[stage+'_delta'],'o-',color=color,label=stage+' original seed 160')
            ax.plot(b.threshold,b[stage+'_delta_median'],'s--',color=color,alpha=.7,label=stage+' repeat median')
            ax.fill_between(b.threshold,b[stage+'_delta_q025'],b[stage+'_delta_q975'],color=color,alpha=.12)
        ax.axhline(0,color='#555',lw=.8);ax.set_xscale('log');ax.set_xticks(THRESHOLDS,THRESHOLDS);ax.set_xlabel('Minimum streamline count per 30,000 seeds');ax.set_ylabel('Post − pre SC–FC Spearman');ax.set_title(support+' SC support; all ranks recomputed');ax.legend(fontsize=7)
    fig.savefig(RESULTS/'figures/sc_repeat_coupling.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,support in zip(axes,['own','common']):
        a=fixed[(fixed.method=='consensus')&(fixed.threshold==1)&(fixed.support==support)].sort_values('consensus_fraction')
        for stage,color in [('primary','#245a81'),('strict','#b35322')]:ax.plot(a.consensus_fraction,a[stage+'_delta'],'o-',color=color,label=stage)
        ax.axhline(0,color='#555',lw=.8);ax.set_xlabel('Minimum fraction of 20 runs with count ≥ 1');ax.set_ylabel('Post − pre SC–FC Spearman');ax.set_title(support+' consensus; mean-count SC weights');ax.legend(fontsize=8)
    fig.savefig(RESULTS/'figures/sc_repeat_consensus.png',dpi=180);plt.close(fig)
