"""Recommended conditional FC stability design; observed fits stay unchanged."""
import json,time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import ROOT,RESULTS,SESSIONS
from .bold import sha
from .censoring import load_session
from .joint_refit import MODES,refit,multiplicities,weighted_correlation,MIN_PAIRED
from .joint_sampling import I,J,B,F,evaluate,spread,variance_components,sc_bootstrap
from .pc_sampling import block_subset,tissue_preprocess,loadings,background_fit,adjusted_overlap,metrics,sample_qc,masks

OUT=RESULTS/'primary_fc'
VERSION='fixed-duration-nonreplacement-refit-v1'
FRACTIONS=('half','two_thirds','four_fifths')
BLOCKS=(5,10,20)
PROTOCOLS=tuple(f'{fraction}_block{block}' for block in BLOCKS for fraction in FRACTIONS)
PRIMARY='two_thirds_block10'


def targets(eligible):
    n=min(map(int,eligible))
    return dict(half=n//2,two_thirds=(2*n)//3,four_fifths=(4*n)//5)


def inputs():
    paths=[ROOT/'src/pat16/primary_fc.py',ROOT/'config/PRIMARY_FC_RESAMPLING.md',RESULTS/'pc_sampling/design.json',RESULTS/'pc_sampling/mechanism_shifts.csv',RESULTS/'nodes.csv',RESULTS/'assessments_all.csv',RESULTS/'evidence_integration/assessment_evidence_ledger.csv']
    paths += [ROOT/'src/pat16'/f'{n}.py' for n in ['pc_sampling','joint_refit','joint_sampling','censoring','nuisance','connectivity','primary_sc']]
    paths += [RESULTS/'joint_sampling'/n for n in ['sc_weights.npy','sc_selected_support.npy','sc_run_indices.npy']]
    for s in SESSIONS:
        paths += [RESULTS/s/n for n in ['roi_timeseries_raw.npy','roi_timeseries_clean.npy','func_motion.tsv','fd.npy','primary_nuisance_pcs.npy','fc_primary_keep.npy','fc_primary.npy']]
        paths += [RESULTS/'censoring_investigation'/s/'tissue_timeseries.npy',RESULTS/'sc_reliability'/s/'counts.npy']
        paths += [RESULTS/'primary_sc'/s/n for n in ['common_support.npy','common_per_10000_seeds.npy','own_per_10000_seeds.npy']]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def analyze(force=False):
    OUT.mkdir(exist_ok=True);lineage=inputs();dp=OUT/'design.json'
    if dp.exists() and not force:
        old=json.loads(dp.read_text())
        if old['inputs']==lineage and all((OUT/p).exists() and sha(OUT/p)==h for p,h in old['outputs'].items()):
            print('Primary FC resampling checkpoint verified',flush=True);return
    started=time.time();eligible=[int(np.load(RESULTS/s/'fc_primary_keep.npy').sum()) for s in SESSIONS]
    counts=targets(eligible);network_masks=masks()
    fc=np.lib.format.open_memmap(OUT/'fc_draws.npy',mode='w+',dtype='float64',shape=(9,3,2,F,len(I)))
    weights=np.zeros((9,2,F,180),np.uint8);tiles={};records=[];qc=[]
    with threadpool_limits(limits=1):
        for p,protocol in enumerate(PROTOCOLS):
            fraction,block=protocol.rsplit('_block',1);block=int(block);target=counts[fraction]
            print('Primary FC design:',protocol,'distinct frames',target,flush=True)
            for t,s in enumerate(SESSIONS):
                d=load_session(s);rng=np.random.default_rng(6160+100*block+10*FRACTIONS.index(fraction)+t)
                origins=[];permutations=[]
                for draw in range(F):
                    idx,origin,order=block_subset(d['primary'],block,target,rng)
                    w=multiplicities(idx,180);weights[p,t,draw]=w;origins.append(origin);permutations.append(order)
                    sq=sample_qc(idx,d['fd']);qc.append(dict(protocol=protocol,session=s,draw=draw,target=target,**sq))
                    bg,x=background_fit(d,w);z,sd,valid=tissue_preprocess(d['tissue'],w);pcs,_,eigen=loadings(z,w)
                    for m,mode in enumerate(MODES):
                        if mode=='frozen':
                            residuals=d['residuals'];fit=weighted_correlation(residuals,w)
                            fitq=dict(eligible=True,design_rank=np.nan,distinct_rows_minus_rank=np.nan)
                        else:fit,fitq,_,residuals=refit(d,idx,d['pcs'] if mode=='coefficients' else pcs)
                        fc[p,m,t,draw]=fit[I,J]
                        diagnostic={}
                        if mode=='full_refit':
                            diagnostic=dict(PCA5_fraction=float(eigen[:5].sum()/eigen.sum()),gap5_relative=float((eigen[4]-eigen[5])/eigen[4]),**adjusted_overlap(d['pcs'],pcs,x,w))
                        records.append(dict(protocol=protocol,fraction=fraction,block=block,mode=mode,session=s,draw=draw,target=target,**fitq,**sq,**diagnostic,**metrics(fit,residuals,bg,w,network_masks)))
                tiles[protocol+'_'+s+'_origins']=np.asarray(origins);tiles[protocol+'_'+s+'_permutations']=np.asarray(permutations)
            fc.flush()
    np.save(OUT/'frame_weights.npy',weights);np.savez_compressed(OUT/'tile_draws.npz',**tiles)
    pd.DataFrame(qc).to_csv(OUT/'sampling_qc.csv',index=False);pd.DataFrame(records).to_csv(OUT/'fit_metrics.csv',index=False)
    summarize(fc);integrate(fc)
    point_fc=np.asarray([np.load(RESULTS/s/'fc_primary.npy')[I,J] for s in SESSIONS])
    point_sc=np.asarray([np.load(RESULTS/'primary_sc'/s/'common_per_10000_seeds.npy')[I,J] for s in SESSIONS])
    support=np.load(RESULTS/'primary_sc/preop/common_support.npy')[I,J]
    point=evaluate(point_sc[:,None,:],support,point_fc[:,None,:])[:,0,0]
    policy=dict(version=VERSION,primary_protocol=PRIMARY,primary_mode='full_refit',primary_support='fixed',eligible_frames=eligible,targets=counts,primary_distinct_frames=counts['two_thirds'],primary_retained_minutes=counts['two_thirds']*2.4/60,primary_block_frames=10,block_seconds=24,SC_draws=B,FC_draws=F,source_sha256=sha(ROOT/'src/pat16/primary_fc.py'),config_sha256=sha(ROOT/'config/PRIMARY_FC_RESAMPLING.md'),decision='Post hoc methodological recommendation; no independent optimality or calibration validation',selection_uses_new_contrast_or_clinical_sign=False,range_label='conditional subset stability range; not a full-run confidence interval',observed_coupling=dict(zip(SESSIONS,map(float,point)))|{'delta':float(point[1]-point[0])})
    (OUT/'policy.json').write_text(json.dumps(policy,indent=2))
    selected=pd.read_csv(OUT/'summary.csv').query("role == 'primary'").iloc[0].to_dict()
    (OUT/'main_summary.json').write_text(json.dumps(dict(policy=policy,primary_joint_coupling=selected,primary_networks='primary_fc/network_uncertainty.csv; role=primary',clinical_evidence='primary_fc/clinical_evidence_ledger.csv',preserved_observed_tables=['summary.json','network_summary.csv','assessments_all.csv'],interpretation='Observed full-run points separate from subset distributions; clinical/context evidence remains descriptive.'),indent=2))
    outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='design.json'}
    dp.write_text(json.dumps(dict(version=VERSION,inputs=lineage,outputs=outputs,protocols=PROTOCOLS,modes=MODES,seconds=time.time()-started),indent=2))
    print('Primary FC resampling completed in',round(time.time()-started,1),'seconds',flush=True)


def summarize(fc):
    sc=np.load(RESULTS/'joint_sampling/sc_weights.npy');reselected=np.load(RESULTS/'joint_sampling/sc_selected_support.npy')
    fixed=np.load(RESULTS/'primary_sc/preop/common_support.npy')[I,J]
    point=np.asarray([np.load(RESULTS/'primary_sc'/s/'common_per_10000_seeds.npy')[I,J] for s in SESSIONS])
    rows=[];sessions=[];feasibility=[];halves=[]
    with threadpool_limits(limits=1):
        for p,protocol in enumerate(PROTOCOLS):
            ids=np.flatnonzero(np.isfinite(fc[p]).all(axis=(0,1,3)));np.save(OUT/f'{protocol}_paired_draws.npy',ids)
            feasibility.append(dict(protocol=protocol,requested=F,paired_eligible=len(ids),summary_available=len(ids)>=MIN_PAIRED))
            if len(ids)<MIN_PAIRED:continue
            deltas=[]
            for m,mode in enumerate(MODES):
                for support in ['fixed']+(['reselected'] if protocol==PRIMARY and mode=='full_refit' else []):
                    mask=fixed if support=='fixed' else reselected
                    rho=evaluate(sc,mask,fc[p,m][:,ids,:]);delta=rho[1]-rho[0]
                    anchor=evaluate(point[:,None,:],fixed,fc[p,m][:,ids,:])[:,0]
                    role='primary' if protocol==PRIMARY and mode=='full_refit' and support=='fixed' else 'sensitivity'
                    base=dict(protocol=protocol,mode=mode,support=support,role=role,SC_draws=B,FC_draws=len(ids))
                    rows.append(base|spread(delta)|variance_components(delta))
                    for t,s in enumerate(SESSIONS):sessions.append(base|{'session':s}|spread(rho[t]))
                    for label,value in [('SC_first',delta[:250]),('SC_second',delta[250:]),('FC_first',delta[:,:len(ids)//2]),('FC_second',delta[:,len(ids)//2:])]:halves.append(base|{'half':label}|spread(value))
                    np.savez_compressed(OUT/f'{protocol}_{support}_{mode}.npz',rho=rho,delta=delta,fc_only_rho=anchor,FC_draw_ids=ids)
                    if support=='fixed':deltas.append(delta)
            np.savez_compressed(OUT/f'{protocol}_paired_shifts.npz',coefficients_minus_frozen=deltas[1]-deltas[0],full_minus_coefficients=deltas[2]-deltas[1],full_minus_frozen=deltas[2]-deltas[0])
    for name,value in [('summary',rows),('session_coupling',sessions),('feasibility',feasibility),('monte_carlo_halves',halves)]:pd.DataFrame(value).to_csv(OUT/f'{name}.csv',index=False)
    catalog=pd.DataFrame(rows);catalog['family']='fixed_common_duration_nonreplacement';catalog['source']='primary_fc/summary.csv'
    old=pd.read_csv(RESULTS/'pc_sampling/summary.csv');old['role']='sensitivity';old['family']='prior_PC_sampling_controls';old['source']='pc_sampling/summary.csv'
    matched=pd.read_csv(RESULTS/'joint_sampling/summary.csv');matched=matched[(matched.source=='joint')&(~matched.protocol.str.contains('block'))].copy();matched['mode']='full_refit';matched['role']='sensitivity';matched['family']='68_frame_duration_FD_run_half_matched';matched['source']='joint_sampling/summary.csv'
    pd.concat([catalog,old,matched],ignore_index=True).to_csv(OUT/'sensitivity_catalog.csv',index=False)


def integrate(fc):
    fixed_masks=masks();network_rows=[];session_rows=[];draws={}
    counts=np.asarray([np.load(RESULTS/'sc_reliability'/s/'counts.npy')[:,I,J] for s in SESSIONS])
    weights,_,hits=sc_bootstrap(counts,np.load(RESULTS/'joint_sampling/sc_run_indices.npy'))
    own=weights*(hits>=16)
    point_sc=np.asarray([np.load(RESULTS/'primary_sc'/s/'own_per_10000_seeds.npy')[I,J] for s in SESSIONS])
    point_fc=np.asarray([np.load(RESULTS/s/'fc_primary.npy')[I,J] for s in SESSIONS])
    for n,(edge_mask,node_mask) in fixed_masks.items():
        sc=own[:,:,edge_mask].mean(-1);obs=point_sc[:,edge_mask].mean(-1);draws[n+'_SC']=sc
        base=dict(network=n,modality='SC',protocol='whole_run_bootstrap',mode='own_reselected',role='primary',units='mean streamlines / 10000 attempted seeds',edges=int(edge_mask.sum()),preop=obs[0],postop=obs[1],observed_delta=obs[1]-obs[0],draws=B)
        network_rows.append(base|spread(sc[1]-sc[0]))
        for t,s in enumerate(SESSIONS):session_rows.append(base|{'session':s,'observed':obs[t]}|spread(sc[t]))
        obs=np.arctanh(np.clip(point_fc[:,edge_mask],-.999999,.999999)).mean(-1)
        for p,protocol in enumerate(PROTOCOLS):
            ids=np.load(OUT/f'{protocol}_paired_draws.npy')
            if len(ids)<MIN_PAIRED:continue
            for m,mode in enumerate(MODES):
                values=np.arctanh(np.clip(fc[p,m][:,ids,:][:,:,edge_mask],-.999999,.999999)).mean(-1)
                role='primary' if protocol==PRIMARY and mode=='full_refit' else 'sensitivity'
                base=dict(network=n,modality='FC',protocol=protocol,mode=mode,role=role,units='mean Fisher z',edges=int(edge_mask.sum()),preop=obs[0],postop=obs[1],observed_delta=obs[1]-obs[0],draws=len(ids))
                network_rows.append(base|spread(values[1]-values[0]))
                for t,s in enumerate(SESSIONS):session_rows.append(base|{'session':s,'observed':obs[t]}|spread(values[t]))
                if role=='primary':draws[n+'_FC']=values
    pd.DataFrame(network_rows).to_csv(OUT/'network_uncertainty.csv',index=False);pd.DataFrame(session_rows).to_csv(OUT/'network_sessions.csv',index=False)
    np.savez_compressed(OUT/'primary_network_draws.npz',**draws)
    ledger=pd.read_csv(RESULTS/'evidence_integration/assessment_evidence_ledger.csv')
    ledger['primary_fc_evidence']='primary_fc/network_uncertainty.csv (role=primary); primary_fc/main_summary.json'
    ledger['primary_fc_inference_limit']='Fixed-duration fully refitted block subsets give conditional stability only; no clinical localization or reliable-change inference; source clinical values unchanged'
    ledger.to_csv(OUT/'clinical_evidence_ledger.csv',index=False)
