import os
from pat16 import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
from pat16.evidence_integration import analyze
analyze()
