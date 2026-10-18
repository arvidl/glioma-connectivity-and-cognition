"""Frozen primary-model identity and shared interactive fitting entry point."""
import numpy as np
from .connectivity import clean_timeseries

VERSION='retained-anatomical5-primary-v1'


def primary_fit(session,fd_threshold=.5,global_signal=None,low_pass=None):
    """Use the actual PCA refit for each censor set, including notebook controls."""
    from .connectivity import functional
    p=functional(session)
    tag={.3:'fd03',.5:'primary',1.:'fd10'}[fd_threshold]
    pcs=np.load(p/f'primary_pcs_{tag}.npy')
    y=np.load(p/'roi_timeseries_raw.npy');motion=np.loadtxt(p/'func_motion.tsv')
    return clean_timeseries(y,motion,pcs,fd_threshold,global_signal,low_pass=low_pass)
