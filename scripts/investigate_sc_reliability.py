"""Repeat fixed-model tractography and evaluate SC/FC sensitivity offline."""
import argparse,os
from threadpoolctl import threadpool_limits
from pat16 import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
p=argparse.ArgumentParser();p.add_argument('--tracking-only',action='store_true');a=p.parse_args()
from pat16.tract_repeats import run
with threadpool_limits(limits=1):
    run()
    if not a.tracking_only:
        from pat16.sc_reliability import run as analyze,plot_results
        analyze();plot_results()
