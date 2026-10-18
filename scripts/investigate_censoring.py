"""Reproduce matched frame-selection experiments on corrected anatomical5 BOLD."""
import argparse, os
from threadpoolctl import threadpool_limits
from pat16 import ROOT
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
from pat16.censoring import run, plot_results
p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=200)
a=p.parse_args()
if a.repeats<2:p.error('--repeats must be at least 2')
with threadpool_limits(limits=1):
    run(a.repeats)
    plot_results()
