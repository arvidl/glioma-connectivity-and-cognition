"""Apply the frozen SC recommendation; retain all historical sensitivity data."""
import os
from pat16 import ROOT
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
from pat16.primary_sc import prepare, analyze
prepare()
analyze()
