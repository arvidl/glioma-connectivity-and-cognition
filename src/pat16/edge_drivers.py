"""Exact rank-product accounting and deletion diagnostics for SC–FC sensitivity.

Edges are dependent, atlas-defined measurements. Contributions are algebraic
bookkeeping in a specified rank universe, not independent effects or importance
probabilities. Recomputed deletion contrasts deliberately change that universe.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from . import ROOT, RESULTS
from .bold import sha
from .censoring import OUT as CENSOR, SESSIONS, STAGES, PROTOCOLS, load_session, fit_selected

OUT=RESULTS/'coupling_edge_drivers'
VERSION='rank-product-edge-drivers-v1'
NETWORKS=['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default']
GROUPS=['shared','pre_only','post_only']
GROUP_LABELS={'shared':'Shared SC','pre_only':'Pre-op-only SC','post_only':'Post-op-only SC'}


def rank_contributions(sc, fc, support=None):
    """Sum of edge contributions equals Spearman, including average-rank ties.

    Absent edges receive zero ACCOUNTING contribution, not zero FC or a place
    in the ranking. Each support subset must be ranked anew for its own rho.
    """
    sc=np.asarray(sc);fc=np.asarray(fc)
    k=(sc>0)&np.isfinite(fc)
    if support is not None:k &= support
    c=np.zeros(sc.shape, float)
    if k.sum()<3:return np.full(sc.shape,np.nan)
    x=rankdata(sc[k],method='average');y=rankdata(fc[k],method='average')
    x-=x.mean();y-=y.mean()
    den=np.sqrt((x@x)*(y@y))
    if den==0:return np.full(sc.shape,np.nan)
    c[k]=x*y/den
    return c


def inputs():
    nodes=pd.read_csv(RESULTS/'nodes.csv')
    i,j=np.triu_indices(len(nodes),1)
    counts=np.array([np.load(RESULTS/s/'sc_counts.npy')[i,j] for s in SESSIONS])
    union=(counts>0).any(0);i,j=i[union],j[union];counts=counts[:,union]
    a,b=nodes.iloc[i].reset_index(drop=True),nodes.iloc[j].reset_index(drop=True)
    group=np.where((counts>0).all(0),'shared',np.where(counts[0]>0,'pre_only','post_only'))
    e=pd.DataFrame({'edge_index':np.arange(len(i)),'node_i':i+1,'node_j':j+1,
                    'label_i':a.label,'label_j':b.label,'network_i':a.network,'network_j':b.network,
                    'pair':[' / '.join(sorted([x,y])) for x,y in zip(a.network,b.network)],
                    'support_group':group,'sc_pre':counts[0],'sc_post':counts[1],
                    'hemisphere_pair':np.where(a.hemisphere==b.hemisphere,a.hemisphere+' / '+b.hemisphere,'LH / RH'),
                    'centroid_distance_mm':np.linalg.norm(a[['x','y','z']].values-b[['x','y','z']].values,axis=1),
                    'min_endpoint_lesion_distance_mm':np.minimum(a.distance_to_lesion_mm,b.distance_to_lesion_mm),
                    'endpoint_overlaps_footprint':(a.lesion_overlap_fraction>0)|(b.lesion_overlap_fraction>0)})
    fc={stage:np.array([np.load(CENSOR/f'{s}_{stage}_fc.npy')[i,j] for s in SESSIONS]) for stage in STAGES}
    if not all(np.isfinite(v).all() for v in fc.values()):
        raise RuntimeError('This fixed-100-node experiment requires complete finite FC')
    return nodes,e,counts,fc


def contrast(counts, fc, removed=None):
    """Recompute all ranks after removing an undirected union-edge set."""
    support=None if removed is None else ~removed
    result={}
    for stage in ['primary','pca']:
        rho=[rank_contributions(s,f,support).sum() for s,f in zip(counts,fc[stage])]
        result[f'{stage}_pre']=rho[0];result[f'{stage}_post']=rho[1]
        result[f'{stage}_delta']=rho[1]-rho[0]
    result['reversal_shift']=result['pca_delta']-result['primary_delta']
    return result


def summarize_edges(e, key):
    rows=[]
    for name,g in e.groupby(key,sort=False):
        row={key:name,'edges':len(g),'positive_edges':int((g.reversal_shift>0).sum()),
             'negative_edges':int((g.reversal_shift<0).sum()),
             'reversal_shift':float(g.reversal_shift.sum()),
             'absolute_shift_sum':float(g.reversal_shift.abs().sum()),
             'shift_per_edge':float(g.reversal_shift.mean()),
             'centroid_distance_median_mm':float(g.centroid_distance_mm.median()),
             'near_footprint_endpoints_le10mm':int((g.min_endpoint_lesion_distance_mm<=10).sum()),
             'footprint_endpoint_edges':int(g.endpoint_overlaps_footprint.sum())}
        for stage in STAGES:row[stage+'_delta_contribution']=float(g[stage+'_delta_contribution'].sum())
        for label in GROUPS:
            row[label+'_edges']=int((g.support_group==label).sum())
            row[label+'_shift']=float(g.loc[g.support_group==label,'reversal_shift'].sum())
        for session,col in [('preop','sc_pre'),('postop','sc_post')]:
            v=g.loc[g[col]>0,col]
            row[session+'_median_count']=float(v.median()) if len(v) else np.nan
            row[session+'_fraction_counts_le2']=float((v<=2).mean()) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def full_sample(e,counts,fc):
    context=[]
    for stage in STAGES:
        c=np.array([rank_contributions(s,f) for s,f in zip(counts,fc[stage])])
        e[stage+'_delta_contribution']=c[1]-c[0]
        for t,session in enumerate(SESSIONS):
            e[f'{session}_{stage}_fc']=fc[stage][t]
            e[f'{session}_{stage}_contribution']=c[t]
            k=counts[t]>0
            x=rankdata(counts[t,k]);y=rankdata(fc[stage][t,k]);x-=x.mean();y-=y.mean()
            denom=np.sqrt((x@x)*(y@y));rank_fraction=(y+(k.sum()+1)/2-1)/(k.sum()-1)
            e[f'{session}_{stage}_fc_rank_fraction']=np.nan
            e.loc[k,f'{session}_{stage}_fc_rank_fraction']=rank_fraction
            for group in GROUPS:
                g=(e.support_group.values[k]==group)
                if not g.any():continue
                between=g.sum()*x[g].mean()*y[g].mean()/denom
                within=np.sum((x[g]-x[g].mean())*(y[g]-y[g].mean()))/denom
                context.append({'session':session,'stage':stage,'support_group':group,'edges':int(g.sum()),
                                'mean_fc':float(fc[stage][t,k][g].mean()),'mean_fc_rank_fraction':float(rank_fraction[g].mean()),
                                'pooled_contribution':float(c[t,k][g].sum()),
                                'within_group_global_rank_term':float(within),'between_group_global_rank_term':float(between),
                                'subgroup_spearman':float(spearmanr(counts[t,k][g],fc[stage][t,k][g]).statistic)})
    e['reversal_shift']=e.pca_delta_contribution-e.primary_delta_contribution
    e['restriction_shift']=e.restriction_delta_contribution-e.primary_delta_contribution
    e['coefficient_refit_shift']=e.coefficients_delta_contribution-e.restriction_delta_contribution
    e['pca_refit_shift']=e.pca_delta_contribution-e.coefficients_delta_contribution
    pd.DataFrame(context).to_csv(OUT/'rank_context.csv',index=False)
    return e


def deletions(e,counts,fc):
    rows=[{'kind':'baseline','removed':'none','union_edges_removed':0,**contrast(counts,fc)}]
    sets=[('support',g,e.support_group.values==g) for g in GROUPS]
    sets += [('support','all_session_specific',e.support_group.values!='shared')]
    sets += [('network_pair',p,e.pair.values==p) for p in sorted(e.pair.unique())]
    sets += [('network_incident',n,(e.network_i.values==n)|(e.network_j.values==n)) for n in NETWORKS]
    for order,label in [(np.argsort(-e.reversal_shift.values),'positive'),
                        (np.argsort(e.reversal_shift.values),'negative'),
                        (np.argsort(-e.reversal_shift.abs().values),'absolute')]:
        for k in [1,5,10,25,50,100]:
            mask=np.zeros(len(e),bool);mask[order[:k]]=True
            sets.append(('top_edges',f'{label}_{k}',mask))
    for kind,name,mask in sets:
        rows.append({'kind':kind,'removed':name,'union_edges_removed':int(mask.sum()),**contrast(counts,fc,mask)})
    pd.DataFrame(rows).to_csv(OUT/'deletions.csv',index=False)
    for idx in range(len(e)):
        mask=np.zeros(len(e),bool);mask[idx]=True
        vals=contrast(counts,fc,mask)
        for key in ['primary_delta','pca_delta','reversal_shift']:e.loc[idx,'leave_one_'+key]=vals[key]
    return e


def matched_accounting(e,counts):
    """Re-evaluate the previous EXACT saved draws; retain per-edge contrasts."""
    arc=np.load(CENSOR/'matched_masks.npz');ds=[load_session(s) for s in SESSIONS]
    i=e.node_i.values-1;j=e.node_j.values-1
    saved=pd.read_csv(CENSOR/'matched_replicates.csv')
    arrays={};checks=[];summary=[]
    partitions={'support':{g:e.support_group.values==g for g in GROUPS},
                'network_pair':{p:e.pair.values==p for p in sorted(e.pair.unique())},
                'network_incident':{n:(e.network_i.values==n)|(e.network_j.values==n) for n in NETWORKS}}
    for protocol in PROTOCOLS:
        masks=arc[protocol]
        for mode in ['frozen','refit']:
            deltas=[]
            reference=saved[(saved.protocol==protocol)&(saved['mode']==mode)].pivot(index='replicate',columns='session',values='coupling')
            for rep,ks in enumerate(masks):
                cs=[]
                for sc,d,k in zip(counts,ds,ks):
                    fc,_,_=fit_selected(d,k,mode)
                    cs.append(rank_contributions(sc,fc[i,j]))
                delta=cs[1]-cs[0];deltas.append(delta)
                error=float(delta.sum()-(reference.loc[rep,'postop']-reference.loc[rep,'preop']))
                if abs(error)>1e-9:raise RuntimeError(f'Matched result differs: {protocol}/{mode}/{rep}: {error}')
                checks.append({'protocol':protocol,'mode':mode,'replicate':rep,'delta':float(delta.sum()),'previous_delta_error':error})
            arrays[protocol+'__'+mode]=np.asarray(deltas)
        arrays[protocol+'__refit_effect']=arrays[protocol+'__refit']-arrays[protocol+'__frozen']
        for mode in ['frozen','refit','refit_effect']:
            z=arrays[protocol+'__'+mode]
            for partition,groups in partitions.items():
                for name,k in groups.items():
                    v=z[:,k].sum(1)
                    summary.append({'protocol':protocol,'mode':mode,'partition':partition,'group':name,'edges':int(k.sum()),
                                    'mean_contribution':float(v.mean()),'median_contribution':float(np.median(v)),
                                    'q025':float(np.quantile(v,.025)),'q975':float(np.quantile(v,.975)),
                                    'positive_fraction':float((v>0).mean())})
        print('Attributed matched draws:',protocol,flush=True)
    np.savez_compressed(OUT/'matched_edge_contrasts.npz',**arrays)
    pd.DataFrame(summary).to_csv(OUT/'matched_group_summary.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'matched_reproduction.csv',index=False)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[RESULTS/'nodes.csv',CENSOR/'design.json',CENSOR/'matched_masks.npz',CENSOR/'matched_replicates.csv']
    for s in SESSIONS:
        sources += [RESULTS/s/n for n in ['sc_counts.npy','sc_fraction.npy','fc_primary.npy','fc_fd03.npy',
                    'roi_timeseries_raw.npy','roi_timeseries_clean.npy','func_motion.tsv','primary_nuisance_pcs.npy']]
        sources += [CENSOR/f'{s}_{stage}_fc.npy' for stage in STAGES]+[CENSOR/s/'tissue_timeseries.npy']
    spec={'version':VERSION,'inputs':{str(p.relative_to(ROOT)):sha(p) for p in sources},
          'code_sha256':sha(ROOT/'src/pat16/edge_drivers.py'),'matched_protocols':list(PROTOCOLS),
          'description':'Own-session nonzero-SC ranks; exact sum accounting, reranked deletion, conditional matched controls'}
    if (OUT/'design.json').exists() and json.loads((OUT/'design.json').read_text())==spec:
        print('Edge attribution cache current',flush=True);return
    nodes,e,counts,fc=inputs()
    e=full_sample(e,counts,fc);e=deletions(e,counts,fc)
    e.to_csv(OUT/'edges.csv',index=False)
    summarize_edges(e,'support_group').to_csv(OUT/'support_summary.csv',index=False)
    summarize_edges(e,'pair').to_csv(OUT/'network_pairs.csv',index=False)
    near=np.where(e.min_endpoint_lesion_distance_mm<=10,'endpoint_le10mm','both_endpoints_gt10mm')
    e['proximity_group']=near
    summarize_edges(e,'proximity_group').to_csv(OUT/'proximity_summary.csv',index=False)
    matched_accounting(e,counts)
    base=contrast(counts,fc)
    summary={'full_sample':base,'union_edges':len(e),'shared_edges':int((e.support_group=='shared').sum()),
             'pre_only_edges':int((e.support_group=='pre_only').sum()),'post_only_edges':int((e.support_group=='post_only').sum()),
             'maximum_single_edge_absolute_shift':float(e.reversal_shift.abs().max()),
             'sum_absolute_edge_shifts':float(e.reversal_shift.abs().sum()),
             'positive_edge_shift_sum':float(e.loc[e.reversal_shift>0,'reversal_shift'].sum()),
             'negative_edge_shift_sum':float(e.loc[e.reversal_shift<0,'reversal_shift'].sum()),
             'single_deletions_preserving_reversal':int(((e.leave_one_primary_delta<0)&(e.leave_one_pca_delta>0)).sum()),
             'matched_scope':'All 6 original protocols × 200 saved draws; frozen/refit; refit_effect uses the same selected frames',
             'interpretation':'Contributions and deletions are descriptive post hoc diagnostics; no independent edge tests or causal biological drivers'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (OUT/'design.json').write_text(json.dumps(spec,indent=2))
    print('Saved edge-driver analysis',flush=True)


def plot_results():
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    e=pd.read_csv(OUT/'edges.csv');g=pd.read_csv(OUT/'support_summary.csv').set_index('support_group').loc[GROUPS]
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(3)
    axs[0].bar(x-.18,g.primary_delta_contribution,width=.35,label='Primary contrast',color='#245a81')
    axs[0].bar(x+.18,g.pca_delta_contribution,width=.35,label='Strict contrast',color='#b35322')
    axs[1].bar(x,g.reversal_shift,color=['#245a81','#c77b32','#8c9298'])
    for ax in axs:
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,[GROUP_LABELS[s] for s in GROUPS],fontsize=9)
    axs[0].set_ylabel('Additive post − pre coupling term');axs[0].legend(fontsize=8)
    axs[1].set_ylabel('Strict − primary contrast term (R)');axs[1].set_ylim(-.006,.028)
    for idx,val in enumerate(g.reversal_shift):axs[1].text(idx,val+(.0007 if val>=0 else -.0025),f'{val:+.4f}',ha='center',fontsize=9)
    axs[0].set_title('A | Global-rank contribution, not subgroup rho');axs[1].set_title('B | Support terms sum to R = +0.0319')
    fig.tight_layout();fig.savefig(RESULTS/'figures/edge_driver_support.png',dpi=180);plt.close(fig)
    pairs=pd.read_csv(OUT/'network_pairs.csv')
    fig,axes=plt.subplots(2,2,figsize=(10.5,8.5),layout='constrained')
    norm=TwoSlopeNorm(vmin=-.013,vcenter=0,vmax=.013)
    for ax,col,title in zip(axes.flat,['reversal_shift','shared_shift','pre_only_shift','post_only_shift'],['All supports','Shared SC','Pre-op-only SC','Post-op-only SC']):
        a=np.zeros((7,7))
        for _,row in pairs.iterrows():
            p,q=[NETWORKS.index(k) for k in row['pair'].split(' / ')]
            a[p,q]=a[q,p]=row[col]
        im=ax.imshow(a,cmap='RdBu_r',norm=norm)
        for p in range(7):
            for q in range(7):
                if p<=q:ax.text(q,p,f'{a[p,q]:+.3f}',ha='center',va='center',fontsize=7,color='white' if abs(a[p,q])>.008 else 'black')
        ax.set_xticks(range(7),NETWORKS,rotation=50,ha='right',fontsize=8);ax.set_yticks(range(7),NETWORKS,fontsize=8);ax.set_title(title,fontsize=11)
    fig.colorbar(im,ax=list(axes.flat),shrink=.65,label='Contribution to strict − primary contrast (R)')
    fig.savefig(RESULTS/'figures/edge_driver_networks.png',dpi=180);plt.close(fig)
    d=pd.read_csv(OUT/'deletions.csv');d=d[d.kind=='network_incident'].set_index('removed').loc[NETWORKS]
    fig,ax=plt.subplots(figsize=(9,4))
    x=np.arange(7);ax.bar(x-.17,d.primary_delta,width=.34,label='Primary after removal',color='#245a81');ax.bar(x+.17,d.pca_delta,width=.34,label='Strict after removal',color='#b35322')
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,NETWORKS,fontsize=9);ax.set_ylabel('Recomputed post − pre coupling');ax.set_xlabel('Network whose incident edges are removed (overlapping sets)');ax.legend(fontsize=8)
    ax.set_title('Deletion reranks the remaining edges; these bars are not additive')
    fig.tight_layout();fig.savefig(RESULTS/'figures/edge_driver_deletions.png',dpi=180);plt.close(fig)


def show_edges(support='all',pair='all',top=10,direction='positive'):
    import matplotlib.pyplot as plt
    e=pd.read_csv(OUT/'edges.csv');nodes=pd.read_csv(RESULTS/'nodes.csv').set_index('id')
    if support!='all':e=e[e.support_group==support]
    if pair!='all':e=e[e.pair==pair]
    if direction=='positive':e=e[e.reversal_shift>0].nlargest(top,'reversal_shift')
    else:e=e[e.reversal_shift<0].nsmallest(top,'reversal_shift')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,coords in zip(axs,[['x','y'],['x','z']]):
        ax.scatter(nodes[coords[0]],nodes[coords[1]],s=9,color='#bbb')
        for _,r in e.iterrows():
            xy=nodes.loc[[r.node_i,r.node_j],coords].values
            ax.plot(xy[:,0],xy[:,1],color='#c36529' if r.reversal_shift>0 else '#245a81',alpha=.7,lw=1.2)
            for idx,point in zip([r.node_i,r.node_j],xy):ax.text(*point,str(idx),fontsize=7)
        ax.set_xlabel('R (mm)');ax.set_ylabel('A (mm)' if coords[1]=='y' else 'S (mm)');ax.set_aspect('equal')
    fig.suptitle('Atlas-centroid schematic in pre-op T1 RAS; straight lines are not tracts',fontsize=10);fig.tight_layout()
    return fig,e
