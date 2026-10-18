import argparse,os
from pat16 import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
from pat16.primary_fc import analyze
p=argparse.ArgumentParser();p.add_argument('--force',action='store_true');a=p.parse_args()
analyze(a.force)
from pat16.primary_fc_figures import figures
figures()
