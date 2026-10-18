"""Recompute BOLD/FC from the saved AP/PA field without refitting diffusion."""
import argparse
from pat16 import RESULTS, SESSIONS
from pat16.bold import correct_bold, variant_folder
from pat16.imaging import rigid_motion_correct
from pat16.connectivity import functional

p=argparse.ArgumentParser()
p.add_argument('--force',action='store_true',help='Invalidate BOLD/FC only; keep DWI topup/eddy and SC')
a=p.parse_args()
for s in SESSIONS:
    if a.force:
        (RESULTS/s/'bold_sdc.json').unlink(missing_ok=True)
        for v in ('corrected','readout_alternative'):(variant_folder(s,v)/'fc_qc.json').unlink(missing_ok=True)
    rigid_motion_correct('func',s)
    correct_bold(s)
    for v in ('uncorrected','corrected','readout_alternative'):functional(s,v)
print('BOLD/FC ready. Refresh all figures, notebooks and report with run_all.py --notebooks --report.')
