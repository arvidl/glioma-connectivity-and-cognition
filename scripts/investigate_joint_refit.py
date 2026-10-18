import argparse,os
from pat16 import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
from pat16.joint_refit import analyze
p=argparse.ArgumentParser();p.add_argument('--force',action='store_true');a=p.parse_args()
analyze(a.force)
