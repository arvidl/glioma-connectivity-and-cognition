"""Allocate coupling sensitivity to edges/supports/networks; no spatial refits."""
import os
from threadpoolctl import threadpool_limits
from pat16 import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
from pat16.edge_drivers import run,plot_results
with threadpool_limits(limits=1):
    run()
    plot_results()
