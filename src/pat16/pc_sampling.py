"""Diagnose PC-training distribution shifts with paired sampling controls.

No new clinical estimates, no replacement of the observed connectomes. Reuse
whole-run SC draws; compare distinct support and multiplicities explicitly.
"""
import json,time
import numpy as np
import pandas as pd
from scipy.linalg import eigh,lstsq,orth,svdvals
from threadpoolctl import threadpool_limits
from . import ROOT,RESULTS,SESSIONS
from .bold import sha
from .censoring import load_session
from .nuisance import temporal_basis
from .joint_refit import OUT as REFIT,MODES,refit,regression_design,weighted_correlation,multiplicities,MIN_PAIRED
from .joint_sampling import OUT as JOINT,I,J,F,B,evaluate,spread,variance_components

OUT=RESULTS/'pc_sampling'
VERSION='pc-training-block-subsampling-v1'
DESIGNS=('bootstrap','unit_coverage','block_unique','block80')
BLOCKS=(5,10,20)
PROTOCOLS=tuple(f'{a}_block{b}' for b in BLOCKS for a in DESIGNS)
MECHANISMS=('coefficients','temporal_only','preprocessing_only','pca_only','full_refit')


def block_subset(keep,block,target,rng):
    """Random circular nonoverlapping tiling; censor inside, trim final tile.

    This is a designed block subset, not uniform sampling of individual frames.
    With target==all eligible frames it returns every eligible frame once.
    """
    keep=np.asarray(keep,bool);nt=len(keep)
    if nt%block or not 0<target<=keep.sum():raise ValueError('Invalid tile size or distinct target')
    origin=int(rng.integers(nt));order=rng.permutation(nt//block)
    tiles=(origin+np.arange(nt))%nt;tiles=tiles.reshape(-1,block)[order]
    idx=tiles.ravel();idx=idx[keep[idx]][:target]
    assert len(np.unique(idx))==target
    return idx,origin,order


def tissue_preprocess(tissue,w,sd_override=None):
    use=w>0;root=np.sqrt(w[use])[:,None];t=temporal_basis(len(tissue))
    z=np.asarray(tissue,float)-t@lstsq(t[use]*root,np.asarray(tissue[use],float)*root,cond=1e-8)[0]
    z-=np.average(z,axis=0,weights=w)
    sd=np.sqrt(np.average(z*z,axis=0,weights=w)) if sd_override is None else sd_override
    valid=sd>1e-5
    return z[:,valid]/sd[valid],sd,valid


def loadings(z,w):
    """Weighted covariance PCA; recentering is absorbed by the GLM intercept."""
    z=z-np.average(z,axis=0,weights=w);use=w>0;zw=z[use]*np.sqrt(w[use,None])
    eigen,u=eigh(zw@zw.T);order=np.argsort(eigen)[::-1]
    eigen=np.maximum(eigen[order],0);u=u[:,order]
    if (eigen>eigen[0]*1e-12).sum()<6:raise RuntimeError('Insufficient PCA rank around 5/6 cutoff')
    v=zw.T@(u[:,:5]/np.sqrt(eigen[:5]))
    return z@v,v,eigen


def background_fit(d,w):
    x=regression_design(d['motion'],np.empty((180,0)),w);use=w>0;root=np.sqrt(w[use,None])
    b=lstsq(x[use]*root,d['y'][use]*root,cond=1e-8)[0]
    return d['y']-x@b,x


def adjusted_overlap(a,b,x,w):
    """PC spans conditional on the identical original-time background GLM."""
    use=w>0;root=np.sqrt(w[use,None]);xx=x[use]*root
    bases=[]
    for y in (a,b):
        yy=y[use]*root;yy-=xx@lstsq(xx,yy,cond=1e-8)[0]
        q=orth(yy,rcond=1e-8)
        if q.shape[1]!=5:raise RuntimeError('Conditional PC span not rank five')
        bases.append(q)
    c=np.clip(svdvals(bases[0].T@bases[1]),0,1)**2
    return dict(adjusted_PC_cos2_min=float(c.min()),adjusted_PC_cos2_mean=float(c.mean()))


def variance(y,w):
    y=y-np.average(y,axis=0,weights=w)
    return np.average(y*y,axis=0,weights=w)


def metrics(fc,residuals,bg,w,masks):
    v=fc[I,J]
    r=dict(mean_fc_r=float(v.mean()),mean_fc_z=float(np.arctanh(np.clip(v,-.999999,.999999)).mean()),
        cortical_mean_PC_removed=float(1-variance(residuals.mean(1),w)/variance(bg.mean(1),w)),
        mean_ROI_PC_removed=float(1-np.mean(variance(residuals,w))/np.mean(variance(bg,w))))
    for n,(edge_mask,node_mask) in masks.items():
        r[n+'_z']=float(np.arctanh(np.clip(v[edge_mask],-.999999,.999999)).mean())
        r[n+'_PC_removed']=float(1-variance(residuals[:,node_mask].mean(1),w)/variance(bg[:,node_mask].mean(1),w))
    return r


def sample_qc(idx,fd):
    w=multiplicities(idx,len(fd));u=np.flatnonzero(w)
    return dict(rows=len(idx),unique=len(u),repeated=len(idx)-len(u),max_multiplicity=int(w.max()),
        frequency_Kish_count=float(w.sum()**2/(w*w).sum()),mean_fd=float(np.average(fd,weights=w)),
        first_half_unique=int((u<90).sum()),adjacent_unique=int((np.diff(u)==1).sum()),
        span_minutes=float((u[-1]-u[0]+1)*2.4/60),unique_minutes=len(u)*2.4/60,
        third1=int((u<60).sum()),third2=int(((u>=60)&(u<120)).sum()),third3=int((u>=120).sum()))


def inputs():
    paths=[ROOT/'src/pat16/pc_sampling.py',ROOT/'config/PC_SAMPLING.md',REFIT/'design.json',REFIT/'fc_draws.npy',JOINT/'design.json',JOINT/'block_indices.npz',JOINT/'sc_weights.npy',JOINT/'sc_selected_support.npy',RESULTS/'nodes.csv',RESULTS/'assessments_all.csv',RESULTS/'evidence_integration/assessment_evidence_ledger.csv']
    paths += [ROOT/'src/pat16'/f'{x}.py' for x in ['joint_sampling','joint_refit','censoring','nuisance','connectivity']]
    for s in SESSIONS:
        paths += [RESULTS/s/n for n in ['roi_timeseries_raw.npy','roi_timeseries_clean.npy','func_motion.tsv','fd.npy','primary_nuisance_pcs.npy','fc_primary_keep.npy','fc_primary.npy']]
        paths += [RESULTS/'censoring_investigation'/s/'tissue_timeseries.npy']
        paths += [RESULTS/'primary_sc'/s/n for n in ['common_support.npy','common_per_10000_seeds.npy']]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def masks():
    ns=pd.read_csv(RESULTS/'nodes.csv').network.to_numpy()
    return {n:((ns[I]==n)&(ns[J]==n),ns==n) for n in dict.fromkeys(ns)}


def analyze(force=False):
    OUT.mkdir(exist_ok=True);lineage=inputs();dp=OUT/'design.json'
    if dp.exists() and not force:
        old=json.loads(dp.read_text())
        if old['inputs']==lineage and all((OUT/p).exists() and sha(OUT/p)==h for p,h in old['outputs'].items()):
            print('PC sampling checkpoint verified',flush=True);return
    started=time.time();saved=np.load(JOINT/'block_indices.npz');network_masks=masks()
    fc=np.lib.format.open_memmap(OUT/'fc_draws.npy',mode='w+',dtype='float64',shape=(12,3,2,F,len(I)))
    fw=np.zeros((12,2,F,180),np.int16);qc=[];records=[];tiles={}
    observed=np.load(REFIT/'fc_draws.npy',mmap_mode='r');max_replay=0.
    with threadpool_limits(limits=1):
        for p,protocol in enumerate(PROTOCOLS):
            design,block=protocol.rsplit('_block',1);block=int(block)
            print('Sampling control:',protocol,flush=True)
            for t,s in enumerate(SESSIONS):
                d=load_session(s);seed=5160+100*block+10*DESIGNS.index(design)+t;rng=np.random.default_rng(seed)
                tile_rows=[];origins=[]
                for f,original in enumerate(saved[f'primary_block{block}_{s}']):
                    idx=original
                    if design=='unit_coverage':idx=np.unique(original)
                    elif design.startswith('block'):
                        target=len(np.unique(original)) if design=='block_unique' else int(.8*d['primary'].sum())
                        idx,origin,order=block_subset(d['primary'],block,target,rng);tile_rows.append(order);origins.append(origin)
                    w=multiplicities(idx,180);fw[p,t,f]=w.astype(np.int16)
                    q=sample_qc(idx,d['fd']);qc.append(dict(protocol=protocol,design=design,block=block,session=s,draw=f,**q))
                    bg,x=background_fit(d,w)
                    z,sd,valid=tissue_preprocess(d['tissue'],w);scores,v,eigen=loadings(z,w)
                    for m,mode in enumerate(MODES):
                        if mode=='frozen':
                            res=d['residuals'];fit=weighted_correlation(res,w);fq={'eligible':True,'design_rank':np.nan,'distinct_rows_minus_rank':np.nan}
                        else:
                            pcs=d['pcs'] if mode=='coefficients' else scores
                            fit,fq,_,res=refit(d,idx,pcs)
                        fc[p,m,t,f]=fit[I,J]
                        extra={}
                        if mode=='full_refit':
                            extra=dict(PCA5_fraction=float(eigen[:5].sum()/eigen.sum()),gap5_relative=float((eigen[4]-eigen[5])/eigen[4]),lambda5_lambda6=float(eigen[4]/eigen[5]),varying_voxels=int(valid.sum()),**adjusted_overlap(d['pcs'],scores,x,w))
                        records.append(dict(protocol=protocol,design=design,block=block,mode=mode,session=s,draw=f,**fq,**q,**extra,**metrics(fit,res,bg,w,network_masks)))
                    if design=='bootstrap':
                        difference=np.max(np.abs(fc[p,:,t,f]-observed[BLOCKS.index(block),:,t,f]))
                        max_replay=max(max_replay,float(difference))
                        if difference>1e-10:raise AssertionError(f'Bootstrap replay differs by {difference}')
                if tile_rows:
                    tiles[protocol+'_'+s+'_permutations']=np.array(tile_rows);tiles[protocol+'_'+s+'_origins']=np.array(origins)
            fc.flush()
    np.save(OUT/'frame_weights.npy',fw);np.savez_compressed(OUT/'tile_draws.npz',**tiles)
    pd.DataFrame(qc).to_csv(OUT/'sampling_qc.csv',index=False)
    pd.DataFrame(records).to_csv(OUT/'fc_metrics.csv',index=False)
    summarize(fc)
    mechanism(network_masks)
    clinical_supplement()
    output={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='design.json'}
    dp.write_text(json.dumps(dict(version=VERSION,inputs=lineage,outputs=output,protocols=PROTOCOLS,modes=MODES,mechanisms=MECHANISMS,SC_draws=B,FC_draws=F,bootstrap_max_absolute_replay_difference=max_replay,seconds=time.time()-started),indent=2))
    print('PC sampling completed in',round(time.time()-started,1),'seconds',flush=True)


def summarize(fc):
    weights=np.load(JOINT/'sc_weights.npy');selected=np.load(JOINT/'sc_selected_support.npy');fixed=np.load(RESULTS/'primary_sc/preop/common_support.npy')[I,J]
    point=np.asarray([np.load(RESULTS/'primary_sc'/s/'common_per_10000_seeds.npy')[I,J] for s in SESSIONS])
    rows=[];sessions=[];feasible=[];halves=[];paired=[]
    with threadpool_limits(limits=1):
        for p,protocol in enumerate(PROTOCOLS):
            ids=np.flatnonzero(np.isfinite(fc[p]).all(axis=(0,1,3)));np.save(OUT/f'{protocol}_paired_draws.npy',ids)
            feasible.append(dict(protocol=protocol,paired_eligible=len(ids),requested=F,summary_available=len(ids)>=MIN_PAIRED))
            if len(ids)<MIN_PAIRED:continue
            ds=[]
            for m,mode in enumerate(MODES):
                supports=['fixed']+(['reselected'] if protocol.endswith('block10') and not protocol.startswith('bootstrap') and mode=='full_refit' else [])
                for support in supports:
                    mask=fixed if support=='fixed' else selected
                    rho=evaluate(weights,mask,fc[p,m][:,ids,:]);delta=rho[1]-rho[0]
                    anchor=evaluate(point[:,None,:],fixed,fc[p,m][:,ids,:])[:,0]
                    base=dict(protocol=protocol,mode=mode,support=support,FC_draws=len(ids),SC_draws=B)
                    rows.append(base|spread(delta)|variance_components(delta))
                    for t,s in enumerate(SESSIONS):sessions.append(base|{'session':s}|spread(rho[t]))
                    for label,a in [('SC_first',delta[:250]),('SC_second',delta[250:]),('FC_first',delta[:,:len(ids)//2]),('FC_second',delta[:,len(ids)//2:])]:halves.append(base|{'half':label}|spread(a))
                    np.savez_compressed(OUT/f'{protocol}_{support}_{mode}.npz',rho=rho,delta=delta,fc_only_rho=anchor,FC_draw_ids=ids)
                    if support=='fixed':ds.append(delta)
            for name,v in [('coefficients_minus_frozen',ds[1]-ds[0]),('full_minus_coefficients',ds[2]-ds[1]),('full_minus_frozen',ds[2]-ds[0])]:paired.append(dict(protocol=protocol,contrast=name)|spread(v))
    for name,v in [('summary',rows),('session_coupling',sessions),('feasibility',feasible),('monte_carlo_halves',halves),('paired_shifts',paired)]:pd.DataFrame(v).to_csv(OUT/f'{name}.csv',index=False)
    metrics_table=pd.read_csv(OUT/'fc_metrics.csv');network_rows=[]
    for protocol in PROTOCOLS:
        for mode in MODES:
            v=metrics_table[(metrics_table.protocol==protocol)&(metrics_table['mode']==mode)]
            for n in masks():
                a=np.asarray([v[v.session==s].sort_values('draw')[n+'_z'].to_numpy() for s in SESSIONS])
                obs=np.asarray([np.load(RESULTS/s/'fc_primary.npy')[I,J] for s in SESSIONS])
                mask=masks()[n][0];op=np.arctanh(np.clip(obs[:,mask],-.999999,.999999)).mean(axis=-1)
                network_rows.append(dict(protocol=protocol,mode=mode,network=n,preop=op[0],postop=op[1],observed_delta=op[1]-op[0],draws=F)|spread(a[1]-a[0]))
    pd.DataFrame(network_rows).to_csv(OUT/'network_ranges.csv',index=False)


def mechanism(network_masks):
    """Paired 2x2 training interventions plus an exact temporal-span control."""
    saved=np.load(JOINT/'block_indices.npz');records=[]
    fcs=np.lib.format.open_memmap(OUT/'mechanism_fc.npy',mode='w+',dtype='float64',shape=(5,2,F,len(I)))
    with threadpool_limits(limits=1):
        for t,s in enumerate(SESSIONS):
            print('Training mechanism controls:',s,flush=True);d=load_session(s)
            baseline_w=d['primary'].astype(float);z0,sd0,valid0=tissue_preprocess(d['tissue'],baseline_w);_,v0,e0=loadings(z0,baseline_w)
            for f,idx in enumerate(saved['primary_block10_'+s]):
                w=multiplicities(idx,180);bg,x=background_fit(d,w)
                z,sd,valid=tissue_preprocess(d['tissue'],w)
                if not np.array_equal(valid,valid0):raise RuntimeError('Changing varying-voxel set: factorial loading controls invalid')
                full,v,e=loadings(z,w);pca,_,ep=loadings(z0,w)
                temporal,_,_=tissue_preprocess(d['tissue'],w,sd_override=sd0)
                all_scores=[d['pcs'],temporal@v0,z@v0,pca,full]
                for m,(mode,scores) in enumerate(zip(MECHANISMS,all_scores)):
                    fc,q,_,res=refit(d,idx,scores);assert q['eligible']
                    fcs[m,t,f]=fc[I,J]
                    eig=ep if mode=='pca_only' else e if mode=='full_refit' else e0
                    records.append(dict(mode=mode,session=s,draw=f,**sample_qc(idx,d['fd']),**q,
                        gap5_relative=float((eig[4]-eig[5])/eig[4]),PCA5_fraction=float(eig[:5].sum()/eig.sum()),
                        **adjusted_overlap(d['pcs'],scores,x,w),**metrics(fc,res,bg,w,network_masks)))
    fcs.flush();pd.DataFrame(records).to_csv(OUT/'mechanism_metrics.csv',index=False)
    weights=np.load(JOINT/'sc_weights.npy');fixed=np.load(RESULTS/'primary_sc/preop/common_support.npy')[I,J]
    rows=[];ds=[]
    with threadpool_limits(limits=1):
        for m,mode in enumerate(MECHANISMS):
            rho=evaluate(weights,fixed,fcs[m]);delta=rho[1]-rho[0];ds.append(delta)
            rows.append(dict(mode=mode)|spread(delta)|variance_components(delta))
            np.savez_compressed(OUT/f'mechanism_{mode}.npz',rho=rho,delta=delta)
    shifts=[]
    pairs=[(1,0,'temporal_minus_coefficients'),(2,0,'preprocessing_minus_coefficients'),(4,2,'full_minus_preprocessing'),(3,0,'pca_minus_coefficients'),(4,3,'full_minus_pca'),(4,0,'full_minus_coefficients')]
    arrays={}
    for a,b,name in pairs:
        v=ds[a]-ds[b];shifts.append(dict(contrast=name)|spread(v));arrays[name]=v
    np.savez_compressed(OUT/'mechanism_shifts.npz',**arrays)
    pd.DataFrame(rows).to_csv(OUT/'mechanism_summary.csv',index=False)
    pd.DataFrame(shifts).to_csv(OUT/'mechanism_shifts.csv',index=False)


def clinical_supplement():
    ledger=pd.read_csv(RESULTS/'evidence_integration/assessment_evidence_ledger.csv')
    ledger['sampling_supplement']='pc_sampling/network_ranges.csv; pc_sampling/summary.csv; observed values unchanged'
    ledger['sampling_interpretation']='Compare all designs; subsample ranges depend on retained fraction; distribution centers do not replace observed changes; no clinical association estimated'
    ledger.to_csv(OUT/'clinical_field_links.csv',index=False)
