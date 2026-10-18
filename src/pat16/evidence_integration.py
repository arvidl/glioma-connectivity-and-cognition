"""Descriptive evidence alignment; no patient-level localization or prediction."""
import json
import numpy as np
import pandas as pd
from . import ROOT,RESULTS,SESSIONS
from .bold import sha
from .clinical import HYPOTHESES
from .joint_refit import OUT as REFIT,PROTOCOLS,MODES
from .joint_sampling import OUT as JOINT,I,J,sc_bootstrap,spread

OUT=RESULTS/'evidence_integration'

def evidence_ledger(a):
    """Preserve every supplied field, unit and caveat, including missingness."""
    a=a.copy();hyp={h[0]:h for h in HYPOTHESES}
    tasks=a.measure.str.split('_').str[0].str.upper()
    a['candidate_networks']=[hyp[t][2] if t in hyp else '' for t in tasks]
    a['link_status']=np.where(a.group=='CANTAB','didactic hypothesis; not individual localization','unlocalized context; no network assigned')
    def role(m):
        if m in ['stai','pswq','loneliness'] or m.startswith('erq_'):return 'reported emotional/social context; potential influence not estimated'
        if m=='employment':return 'participation/context; no dedicated quality-of-life measurement'
        return 'background or behavioral context; coding and timing limits retained'
    a['integration_role']=[hyp[t][1] if t in hyp else role(m) for t,m in zip(tasks,a.measure)]
    a['availability']=[('paired' if pd.notna(p) and pd.notna(q) else 'pre-only' if pd.notna(p) else 'post-only' if pd.notna(q) else 'missing') for p,q in zip(a.preop,a.postop)]
    a['connectivity_evidence']='network_summary.csv; network_uncertainty.csv; joint_refit/summary.csv; original joint_sampling matched designs'
    a['inference_limit']='raw descriptive clinical change only; no reliable-change or causal/individual network estimate'
    return a

def analyze():
    OUT.mkdir(exist_ok=True)
    a=pd.read_csv(RESULTS/'assessments_all.csv');ledger=evidence_ledger(a)
    ledger.to_csv(OUT/'assessment_evidence_ledger.csv',index=False)
    nodes=pd.read_csv(RESULTS/'nodes.csv');networks=list(dict.fromkeys(nodes.network));labels=nodes.network.to_numpy()
    masks=[(labels[I]==n)&(labels[J]==n) for n in networks]
    ids=np.load(REFIT/'primary_block10_paired_draws.npy')
    f=np.load(REFIT/'fc_draws.npy',mmap_mode='r')[PROTOCOLS.index('primary_block10')]
    point_fc=np.asarray([np.load(RESULTS/s/'fc_primary.npy')[I,J] for s in SESSIONS])
    point_sc=np.asarray([np.load(RESULTS/'primary_sc'/s/'own_per_10000_seeds.npy')[I,J] for s in SESSIONS])
    counts=np.asarray([np.load(RESULTS/'sc_reliability'/s/'counts.npy')[:,I,J] for s in SESSIONS])
    w,_,hits=sc_bootstrap(counts,np.load(JOINT/'sc_run_indices.npy'))
    own=w*(hits>=16)
    rows=[];session_rows=[];draw_arrays={}
    for n,mask in zip(networks,masks):
        sc=own[:,:,mask].mean(axis=-1);sp=point_sc[:,mask].mean(axis=-1)
        draw_arrays[n+'_SC']=sc
        rows.append(dict(network=n,modality='SC',mode='own_reselected',draws=sc.shape[1],edges=int(mask.sum()),units='mean streamlines / 10000 attempted seeds',preop=sp[0],postop=sp[1],observed_delta=sp[1]-sp[0])|spread(sc[1]-sc[0]))
        for m,mode in enumerate(MODES):
            v=np.arctanh(np.clip(f[m][:,ids,:][:,:,mask],-.999999,.999999)).mean(axis=-1)
            p=np.arctanh(np.clip(point_fc[:,mask],-.999999,.999999)).mean(axis=-1)
            draw_arrays[n+'_'+mode]=v
            base=dict(network=n,modality='FC',mode=mode,draws=len(ids),edges=int(mask.sum()),units='mean Fisher z',preop=p[0],postop=p[1],observed_delta=p[1]-p[0])
            rows.append(base|spread(v[1]-v[0]))
            for t,s in enumerate(SESSIONS):session_rows.append(base|{'session':s}|spread(v[t]))
        for t,s in enumerate(SESSIONS):session_rows.append(dict(network=n,modality='SC',mode='own_reselected',draws=sc.shape[1],session=s)|spread(sc[t]))
    pd.DataFrame(rows).to_csv(OUT/'network_uncertainty.csv',index=False)
    pd.DataFrame(session_rows).to_csv(OUT/'network_sessions.csv',index=False)
    np.savez_compressed(OUT/'network_draws.npz',**draw_arrays,FC_draw_ids=ids)
    pd.DataFrame(HYPOTHESES,columns=['task','domain','candidate_networks','interpretive_limit']).to_csv(OUT/'task_network_hypotheses.csv',index=False)
    figures()
    paths=[ROOT/'src/pat16/evidence_integration.py',ROOT/'src/pat16/clinical.py',ROOT/'config/JOINT_REFIT.md',RESULTS/'assessments_all.csv',RESULTS/'nodes.csv',REFIT/'design.json',REFIT/'primary_block10_paired_draws.npy',REFIT/'fc_draws.npy',JOINT/'sc_run_indices.npy']
    paths += [RESULTS/'sc_reliability'/s/'counts.npy' for s in SESSIONS]
    paths += [RESULTS/'primary_sc'/s/'own_per_10000_seeds.npy' for s in SESSIONS]
    paths += [RESULTS/s/'fc_primary.npy' for s in SESSIONS]
    (OUT/'design.json').write_text(json.dumps(dict(version='descriptive-network-clinical-evidence-v1',fields=len(ledger),cantab_fields=int((ledger.group=='CANTAB').sum()),context_fields=int((ledger.group!='CANTAB').sum()),inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='design.json'},interpretation='Conditional computational ranges alongside raw clinical changes; no correlations, composites, reliable-change tests or individual localization.'),indent=2))
    print('Integrated',len(ledger),'clinical/context fields and',len(networks),'networks')

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=pd.read_csv(OUT/'network_uncertainty.csv')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    v=d[d.modality=='SC'];y=np.arange(len(v))
    axs[0].errorbar(v['median'],y,xerr=np.array([v['median']-v.q025,v.q975-v['median']]),fmt='o',capsize=3,color='#377d78')
    axs[0].plot(v.observed_delta,y,'x',color='black',label='Observed Δ');axs[0].set_yticks(y,v.network);axs[0].set_title('SC: own consensus reselected in each run bootstrap');axs[0].set_xlabel('Post − pre mean streamlines / 10,000 seeds')
    for offset,mode,color in [(-.18,'frozen','#4b8c89'),(0,'coefficients','#d59c3a'),(.18,'full_refit','#425b9a')]:
        v=d[(d.modality=='FC')&(d['mode']==mode)]
        axs[1].errorbar(v['median'],y+offset,xerr=np.array([v['median']-v.q025,v.q975-v['median']]),fmt='o',capsize=2,color=color,label={'frozen':'Frozen','coefficients':'Refit coefficients','full_refit':'Refit PCs + coefficients'}[mode],ms=4)
    axs[1].plot(v.observed_delta,y,'x',color='black',label='Observed Δ');axs[1].set_yticks(y,v.network);axs[1].set_title('FC: paired primary block-10 resamples');axs[1].set_xlabel('Post − pre within-network mean Fisher z')
    for ax in axs:ax.axvline(0,color='gray',ls='--');ax.invert_yaxis()
    fig.legend(*axs[1].get_legend_handles_labels(),loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('Network summaries for descriptive clinical alignment; conditional 2.5/97.5 percentiles')
    fig.savefig(RESULTS/'figures/evidence_network_uncertainty.png',dpi=180);plt.close(fig)
