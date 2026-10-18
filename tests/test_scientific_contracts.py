import numpy as np
import pytest
from pat16.connectivity import endpoints_to_matrix, fd_from_motion, clean_timeseries
from pat16.networks import proportional_threshold,metrics,attenuate,ou_fc,graph_growth,network_means


def test_endpoint_assignment_radius_and_undirected_counts():
    atlas=np.zeros((10,3,3),int);atlas[1,1,1]=1;atlas[8,1,1]=2
    sl=[np.array([[1,1,1],[8,1,1]]),np.array([[1,1,1],[1,1,1]]),np.array([[1,1,1],[80,1,1]])]
    c,p,d=endpoints_to_matrix(sl,atlas,np.eye(4),2)
    assert c[0,1]==c[1,0]==1 and c.sum()==2
    assert (p[1:]==-1).all()


def test_threshold_never_invents_edges_and_handles_ties():
    w=np.array([[0,1,1],[1,0,0],[1,0,0.]])
    t=proportional_threshold(w,.34)
    assert (t==t.T).all() and np.count_nonzero(t)==2
    assert np.count_nonzero(proportional_threshold(w,1))==4
    assert metrics(np.zeros((3,3)))['weighted_global_efficiency']==0


def test_ou_lyapunov_uncoupled_and_injury_monotonicity():
    w=np.ones((5,5))-np.eye(5)
    assert np.allclose(ou_fc(w,0),np.eye(5))
    fc=ou_fc(w,.7); assert np.allclose(fc,fc.T) and np.allclose(np.diag(fc),1)
    hurt=attenuate(w,np.array([1,0,0,0,0]),1)
    assert np.all(hurt[0]==0)
    assert metrics(hurt,1,scale=1)['weighted_global_efficiency'] < metrics(w,1,scale=1)['weighted_global_efficiency']
    with pytest.raises(ValueError):ou_fc(w,1)


def test_graph_growth_has_no_spontaneous_mass():
    h=graph_growth(np.zeros((4,4)),0)
    assert (h[:,1:]==0).all() and ((h>=0)&(h<=1)).all()


def test_motion_fd_converts_rotation_to_mm():
    m=np.zeros((3,6));m[1,0]=.01;m[1,3]=1
    assert fd_from_motion(m)[1]==pytest.approx(1.5)


def test_fc_keeps_missing_nodes_and_censors_on_original_time_axis():
    rng=np.random.default_rng(42);y=rng.normal(size=(180,4));y[:,3]=np.nan
    motion=np.zeros((180,6));motion[60,3]=2
    fc,res,keep,q,fd=clean_timeseries(y,motion,rng.normal(size=(180,5)))
    assert not keep[60] and not keep[61] and not keep[:4].any()
    assert q['residual_df']>15
    assert np.isnan(fc[0,3]) and np.isfinite(fc[:3,:3]).all()


def test_network_mean_excludes_self_connections():
    w=np.eye(4);w[0,1]=w[1,0]=.2
    m,n=network_means(w,['a','a','b','b'])
    assert m[0,0]==pytest.approx(.2)
