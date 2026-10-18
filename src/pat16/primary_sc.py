"""Reliability-based SC policy, independent of FC/cognitive outcome selection.

Session consensus describes topology. Its fixed paired intersection defines the
longitudinal coupling population. Original reconstructions remain immutable.
"""
import json
import numpy as np
import pandas as pd
from . import ROOT, RESULTS, SESSIONS
from .bold import sha
from .networks import metrics, network_means, sc_fc_coupling

OUT = RESULTS/'primary_sc'
VERSION = 'repeat80-mean-seed-rate-common-coupling-v1'
REPEATS, SEEDS, MIN_HITS = 20, 30000, 16


def policy_arrays(counts):
    """Counts: session x repeat x node x node; average includes zero runs."""
    if counts.shape != (2, REPEATS, 100, 100):
        raise ValueError('Expected two sessions, twenty repeats and 100 nodes')
    hits = (counts >= 1).sum(axis=1)
    own = hits >= MIN_HITS
    common = own.all(axis=0)
    mean = counts.mean(axis=1)
    return mean, hits, own, common


def load_weights(session, support='own'):
    if session not in SESSIONS or support not in ['own', 'common']:
        raise ValueError('Unknown session or SC support')
    return np.load(OUT/session/f'{support}_per_10000_seeds.npy')


def prepare():
    """Build SC matrices without reading FC, cognition or coupling results."""
    counts = np.array([np.load(RESULTS/'sc_reliability'/s/'counts.npy') for s in SESSIONS])
    mean, hits, own, common = policy_arrays(counts)
    spec = {'version': VERSION, 'post_hoc_methodological_recommendation': True,
            'repeats': REPEATS, 'seeds_per_repeat': SEEDS, 'minimum_hits': MIN_HITS,
            'per_run_inclusive_count_cutoff': 1,
            'weight': 'arithmetic mean of all 20 counts, including zeros, per 10000 attempted seeds',
            'graph_network_simulation_support': 'session-specific 16/20 consensus',
            'longitudinal_coupling_support': 'intersection of session consensus supports and common finite FC nodes',
            'selection_uses_FC_or_cognition': False, 'policy_choice_blinded_or_preregistered': False,
            'limitations': 'conditional seed stability only; no anatomical validity or scan-rescan calibration; post-op data enter paired support',
            'inputs': {s: sha(RESULTS/'sc_reliability'/s/'counts.npy') for s in SESSIONS},
            'source_sha256': sha(__file__), 'config_sha256': sha(ROOT/'config/PRIMARY_SC.md'),
            'sessions': {}}
    for t, s in enumerate(SESSIONS):
        p = OUT/s; p.mkdir(parents=True, exist_ok=True)
        np.save(p/'mean_counts.npy', mean[t]); np.save(p/'detection_hits.npy', hits[t])
        np.save(p/'own_support.npy', own[t]); np.save(p/'common_support.npy', common)
        for name, support in [('own', own[t]), ('common', common)]:
            np.save(p/f'{name}_per_10000_seeds.npy', np.where(support, mean[t]*10000/SEEDS, 0.))
        spec['sessions'][s] = {'consensus_edges': int(np.triu(own[t], 1).sum()),
            'common_edges': int(np.triu(common, 1).sum()),
            'session_specific_consensus_edges': int(np.triu(own[t] & ~common, 1).sum()),
            'retained_mean_assignments_per_30000_seeds': float(np.triu(np.where(own[t], mean[t], 0.), 1).sum())}
    (OUT/'policy.json').write_text(json.dumps(spec, indent=2))
    return spec


def analyze():
    """Re-evaluate coupling, graph/network sensitivities and saved FC fits."""
    from .sc_reliability import load, consensus_weights, threshold_weights, evaluate, THRESHOLDS, CONSENSUS
    from .edge_drivers import rank_contributions
    from .connectivity import clean_timeseries
    i, j, baseline, repeats, fc = load()
    nodes = pd.read_csv(RESULTS/'nodes.csv'); groups = nodes.network.to_numpy()
    valid = np.load(RESULTS/'common_fc_nodes.npy'); valid_edges = valid[i] & valid[j]
    designs = [('original_fraction', 'own', np.array([np.load(RESULTS/s/'sc_fraction.npy')[i,j] for s in SESSIONS]))]
    for support in ['own', 'common']:
        for k in THRESHOLDS:
            designs.append((f'original_k{k}', support, threshold_weights(baseline, k, support)*10000/SEEDS))
        designs.append(('repeat_mean_unfiltered', support, threshold_weights(repeats.mean(1), np.nextafter(0.,1.), support)*10000/SEEDS))
        for k in THRESHOLDS:
            for p in CONSENSUS:
                designs.append((f'consensus{int(p*100)}_k{k}', support, consensus_weights(repeats, k, p, support)*10000/SEEDS))
    coupling, graphs, nets = [], [], []
    for name, support, w in designs:
        row = {'design': name, 'support': support,
               'coupling_role': 'primary' if name=='consensus80_k1' and support=='common' else 'sensitivity',
               'structural_role': 'primary' if name=='consensus80_k1' and support=='own' else 'sensitivity'}
        coupling.append({**row, **evaluate(np.where(valid_edges, w, 0.), fc)})
        matrices = np.zeros((2, 100, 100)); matrices[:,i,j] = w; matrices[:,j,i] = w
        scale = matrices.max()
        for t, s in enumerate(SESSIONS):
            for density in [.1,.15,.2,.25,.3]:
                graphs.append({**row, 'session':s, 'target_density':density, **metrics(matrices[t],density,scale)})
            net, names = network_means(matrices[t], groups)
            for a, network in enumerate(names):
                nets.append({**row, 'session':s, 'network':network, 'within_sc_weight':net[a,a]})
    pd.DataFrame(coupling).to_csv(OUT/'coupling_sensitivity.csv', index=False)
    pd.DataFrame(graphs).to_csv(OUT/'graph_sensitivity.csv', index=False)
    pd.DataFrame(nets).to_csv(OUT/'network_sensitivity.csv', index=False)
    # All four ordered FC stages are re-ranked on the same primary SC population.
    stages = []; terms = pd.DataFrame({'node_i':i+1,'node_j':j+1})
    terms['network_pair'] = [' / '.join(sorted([a,b])) for a,b in zip(groups[i],groups[j])]
    weights = np.array([load_weights(s,'common')[i,j] for s in SESSIONS])
    weights[:,~valid_edges] = 0.
    selected = (weights>0).all(0); terms['primary_coupling_support'] = selected
    for stage in ['primary','restriction','coefficients','pca']:
        rho = []
        for t, s in enumerate(SESSIONS):
            f = np.load(RESULTS/'censoring_investigation'/f'{s}_{stage}_fc.npy')[i,j]
            c = rank_contributions(weights[t], f); terms[f'{s}_{stage}_contribution'] = c
            rho.append(c.sum())
        stages.append({'stage':stage,'pre_rho':rho[0],'post_rho':rho[1],'delta':rho[1]-rho[0],'edges':int(selected.sum())})
    terms['primary_delta_term'] = terms.postop_primary_contribution-terms.preop_primary_contribution
    terms['strict_delta_term'] = terms.postop_pca_contribution-terms.preop_pca_contribution
    terms['reversal_shift_term'] = terms.strict_delta_term-terms.primary_delta_term
    terms[selected].to_csv(OUT/'edge_contributions.csv',index=False)
    terms[selected].groupby('network_pair').agg(edges=('node_i','size'),primary_delta_term=('primary_delta_term','sum'),strict_delta_term=('strict_delta_term','sum'),reversal_shift_term=('reversal_shift_term','sum')).reset_index().to_csv(OUT/'network_contributions.csv',index=False)
    pd.DataFrame(stages).to_csv(OUT/'censoring_decomposition.csv',index=False)
    # Saved complete nuisance refits on all 6x200 frame masks already used this SC.
    for name in ['matched_coupling','matched_coupling_summary']:
        d = pd.read_csv(RESULTS/'sc_reliability'/f'{name}.csv')
        d[d.sc_reference=='consensus80_common'].to_csv(OUT/f'{name}.csv',index=False)
    den = pd.read_csv(RESULTS/'nuisance_denoising_comparison.csv')
    for idx, row in den.iterrows():
        f = np.load(RESULTS/'nuisance_investigation'/row.session/row.BOLD_processing/f'fc_{row.model}.npy')
        den.loc[idx,'sc_fc_spearman'] = sc_fc_coupling(load_weights(row.session,'common')[np.ix_(valid,valid)], f)
    den['SC_reference'] = VERSION; den.to_csv(OUT/'denoising_comparison.csv',index=False)
    bc = pd.read_csv(RESULTS/'bold_comparison.csv')
    for idx, row in bc.iterrows():
        p = (RESULTS/'nuisance_investigation'/row.session/'uncorrected'/'fc_anatomical5.npy' if row.BOLD_processing=='uncorrected' else
             RESULTS/row.session/'fc_primary.npy' if row.BOLD_processing=='corrected' else RESULTS/'bold_readout_alternative'/row.session/'fc_primary.npy')
        f = np.load(p)
        if len(f)==100: f = f[np.ix_(valid,valid)]
        bc.loc[idx,'sc_fc_spearman'] = sc_fc_coupling(load_weights(row.session,'common')[np.ix_(valid,valid)], f)
    bc['SC_reference'] = VERSION; bc.to_csv(OUT/'bold_comparison.csv',index=False)
    adjacent = pd.read_csv(RESULTS/'primary_pc_sensitivity.csv')
    for s in SESSIONS:
        for variant in ['corrected','uncorrected']:
            p = RESULTS/s if variant=='corrected' else RESULTS/'bold_uncorrected'/s
            y = np.load(p/'roi_timeseries_raw.npy'); motion = np.loadtxt(p/'func_motion.tsv')
            pcs = np.load(RESULTS/'nuisance_investigation'/s/variant/'anatomical_pcs20.npy')
            for k in range(21):
                f = clean_timeseries(y,motion,pcs[:,:k])[0][np.ix_(valid,valid)]
                mask = (adjacent.session==s)&(adjacent.BOLD_processing==variant)&(adjacent.components==k)
                adjacent.loc[mask,'sc_fc_spearman'] = sc_fc_coupling(load_weights(s,'common')[np.ix_(valid,valid)],f)
    adjacent['SC_reference'] = VERSION; adjacent.to_csv(OUT/'pc_sensitivity.csv',index=False)
    figures(pd.DataFrame(coupling), pd.DataFrame(stages), den, bc, adjacent)
    print('Primary SC policy and all structural/coupling sensitivity tables updated',flush=True)


def figures(coupling, stages, den, bc, adjacent):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(1,2,figsize=(11,4))
    selected = coupling[((coupling.design=='original_k1')&(coupling.support=='own'))|
                        ((coupling.design=='consensus80_k1'))]
    x = np.arange(len(selected)); labels = ['Original own','Consensus own','Consensus common']
    for ax,col,title in zip(axs,['primary_delta','strict_delta'],['FD 0.5','FD 0.3, full refit']):
        ax.bar(x,selected[col],color=['#999999','#d4952e','#328c91']);ax.set_xticks(x,labels,rotation=12)
        ax.axhline(0,color='.4',lw=.7);ax.set_title(title);ax.set_ylabel('Post minus pre SC–FC Spearman')
    fig.suptitle('SC population comparison | common consensus is primary for paired coupling')
    fig.tight_layout();fig.savefig(RESULTS/'figures/primary_sc_choice.png',dpi=180);plt.close(fig)
    for table,columns,titles,filename,xcol in [
        (bc,['mean_fc_r','sc_fc_spearman','weighted_global_efficiency'],['Mean FC r','Coupling, common consensus','Positive-FC efficiency'],'primary_sc_bold','session'),
        (adjacent[adjacent.BOLD_processing=='corrected'],['mean_fc_r','sc_fc_spearman'],['Mean FC r','Coupling, common consensus'],'primary_sc_pc_count','components'),
        (den[den.BOLD_processing=='corrected'],['mean_fc_r','sc_fc_spearman'],['Mean FC r','Coupling, common consensus'],'primary_sc_denoising','model')]:
        fig,axs = plt.subplots(1,len(columns),figsize=(11,3.8))
        for ax,col,title in zip(axs,columns,titles):
            if xcol=='session':
                for variant,d in table.groupby('BOLD_processing',sort=False):ax.plot(d[xcol],d[col],'o-',label=variant)
            else:
                for s,d in table.groupby('session',sort=False):ax.plot(d[xcol],d[col],'o-',label=s,ms=3)
            ax.set_title(title);ax.tick_params(axis='x',labelrotation=30);ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(RESULTS/'figures'/f'{filename}.png',dpi=180);plt.close(fig)
