"""Network definitions, sensitivity analysis and teaching simulations."""
import numpy as np
import networkx as nx
from scipy.linalg import solve_continuous_lyapunov, expm
from scipy.stats import spearmanr


def proportional_threshold(weights,density):
    w=np.array(weights,float,copy=True)
    if not 0<=density<=1:raise ValueError('Density must lie in [0,1]')
    w[~np.isfinite(w)]=0;w=np.maximum((w+w.T)/2,0);np.fill_diagonal(w,0)
    i,j=np.triu_indices(len(w),1);positive=np.where(w[i,j]>0)[0]
    k=min(int(round(density*len(i))),len(positive))
    # Stable tie ordering, no artificial edges when fewer exist.
    order=positive[np.argsort(-w[i[positive],j[positive]],kind='stable')[:k]]
    out=np.zeros_like(w);out[i[order],j[order]]=w[i[order],j[order]];return out+out.T


def metrics(weights,density=.2,scale=None):
    w=proportional_threshold(weights,density)
    if scale is None:scale=np.nanmax(w) if np.any(w) else 1.
    G=nx.from_numpy_array(w/max(scale,1e-12))
    for u,v in G.edges:G[u][v]['distance']=1/G[u][v]['weight']
    lengths=dict(nx.all_pairs_dijkstra_path_length(G,weight='distance'))
    n=len(w);eff=sum(1/d for a,ds in lengths.items() for b,d in ds.items() if a!=b and d>0)/(n*(n-1))
    degree=(w>0).sum(1);triangles=np.diag(np.linalg.matrix_power(np.cbrt(w/max(scale,1e-12)),3));clustering=np.divide(triangles,degree*(degree-1),out=np.zeros(n),where=degree>1)
    return {'nodes':n,'edges':G.number_of_edges(),'density':nx.density(G),'components':nx.number_connected_components(G),'largest_component':len(max(nx.connected_components(G),key=len)),'binary_global_efficiency':nx.global_efficiency(G),'weighted_global_efficiency':eff,'weighted_clustering':float(clustering.mean()),'mean_strength':float(w.sum(1).mean()),'weight_scale':float(scale)}


def sc_fc_coupling(sc,fc):
    i,j=np.triu_indices(len(sc),1)
    valid=np.isfinite(fc[i,j])&(sc[i,j]>0)
    # Descriptive edge-level rank correlation; edges are dependent, no p-value.
    if valid.sum()<3:return np.nan
    return float(spearmanr(np.log1p(sc[i,j][valid]),fc[i,j][valid]).statistic)


def network_means(matrix,groups,fisher=False):
    names=list(dict.fromkeys(groups));out=np.full((len(names),len(names)),np.nan)
    x=np.array(matrix,copy=True,dtype=float)
    if fisher:x=np.arctanh(np.clip(x,-.999999,.999999))
    np.fill_diagonal(x,np.nan)
    for a,na in enumerate(names):
        for b,nb in enumerate(names):
            block=x[np.ix_(np.array(groups)==na,np.array(groups)==nb)]
            if np.isfinite(block).any():out[a,b]=np.nanmean(block)
    return out,names


def attenuate(sc,exposure,severity):
    """Hypothetical edge injury: diagonal node survival factors on both sides."""
    survival=np.clip(1-severity*np.asarray(exposure),0,1)
    return survival[:,None]*sc*survival[None,:]


def ou_fc(sc,gain=.6,scale=None):
    """Stationary OU covariance; stable for gain<1 with baseline spectral scaling."""
    if not 0<=gain<1:raise ValueError('OU gain must be below 1 for guaranteed stability')
    rho=max(float(np.linalg.eigvalsh(sc).max()),1e-12) if scale is None else scale
    A=-np.eye(len(sc))+gain*sc/rho
    if np.linalg.eigvalsh(A).max()>=0:raise ValueError('Unstable dynamics')
    cov=solve_continuous_lyapunov(A,-np.eye(len(sc)))
    sd=np.sqrt(np.diag(cov));return cov/(sd[:,None]*sd[None,:])


def graph_growth(sc,seed,steps=50,dt=.1,diffusion=.5,rho=.2):
    """Didactic Fisher-KPP on a graph; concentration is entirely simulated."""
    w=sc/max(np.linalg.eigvalsh(sc).max(),1e-12);L=np.diag(w.sum(1))-w
    x=np.zeros(len(w));x[seed]=.1;history=[x.copy()]
    for _ in range(steps):
        x=np.clip(x+dt*(-diffusion*L@x+rho*x*(1-x)),0,1);history.append(x.copy())
    return np.array(history)


def block_bootstrap_network(series,keep,groups,seed=160,repeats=200,block=10):
    """Moving blocks chosen on original time axis, retaining gaps within blocks.

    Intervals describe within-run sampling instability conditional on nuisance fit.
    They are not surgery-effect intervals or independent-observation tests.
    """
    rng=np.random.default_rng(seed);n=len(series);summaries=[]
    valid=np.isfinite(series).all(0)
    for _ in range(repeats):
        idx=np.concatenate([np.arange(s,s+block)%n for s in rng.integers(0,n,size=int(np.ceil(n/block)))])[:n]
        idx=idx[keep[idx]]
        f=np.full((len(groups),len(groups)),np.nan)
        f[np.ix_(valid,valid)]=np.corrcoef(series[idx][:,valid],rowvar=False)
        a,_=network_means(f,groups,fisher=True);summaries.append(np.diag(a))
    return np.nanpercentile(summaries,[2.5,50,97.5],axis=0)
