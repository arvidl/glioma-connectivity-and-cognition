"""Reproduce conditional joint SC/FC uncertainty from bundled derivatives."""
import argparse, os
from pat16 import ROOT
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
from pat16.joint_sampling import analyze
p=argparse.ArgumentParser()
p.add_argument('--force',action='store_true')
a=p.parse_args()
analyze(force=a.force)
