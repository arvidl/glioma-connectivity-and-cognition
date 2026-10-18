import argparse,os
from pat16 import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
from pat16.pc_sampling import analyze
p=argparse.ArgumentParser();p.add_argument('--force',action='store_true');a=p.parse_args()
analyze(a.force)
from pat16.pc_sampling_figures import figures
figures()
