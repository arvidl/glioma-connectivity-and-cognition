"""Single entry point. Recompute everything within this directory."""
import argparse, subprocess, sys, time
from pathlib import Path
from pat16 import ROOT,RESULTS,SESSIONS
import os
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
p=argparse.ArgumentParser();p.add_argument('--from-raw',action='store_true',help='Register images, realign volumes and reconstruct paired connectomes');p.add_argument('--force',action='store_true',help='Invalidate computed checkpoints (requires --from-raw)');p.add_argument('--notebooks',action='store_true');p.add_argument('--report',action='store_true');a=p.parse_args()
if a.force and not a.from_raw:p.error('--force requires --from-raw')
if a.force:
    for f in [RESULTS/'anatomy/complete.json']+[RESULTS/s/name for s in SESSIONS for name in ['dwi_correction.json','correction/topup_complete.json','correction/eddy_complete.json','func_motion.json','bold_sdc.json','sc_qc.json','fc_qc.json']]:f.unlink(missing_ok=True)
    for s in SESSIONS:
        for name in ['dwi_motion.json','sc_qc.json']:(RESULTS/'uncorrected_matched'/s/name).unlink(missing_ok=True)
start=time.time()
subprocess.run([sys.executable,str(ROOT/'scripts/manifest.py'),'--verify'],check=True,cwd=ROOT)
if a.from_raw:
    from pat16.imaging import anatomy,motion_correct
    from pat16.connectivity import structural,functional
    anatomy()
    if a.force:
        from pat16.nuisance import segment
        for s in SESSIONS:segment(s,force=True)
    for s in SESSIONS:
        motion_correct('dwi',s)
        structural(s)
        motion_correct('func',s)
        functional(s)
from pat16.connectivity import functional
for s in SESSIONS:functional(s)
for script in ['correction_comparison.py','correction_qc.py','bold_qc.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts'/script)],check=True,cwd=ROOT)
# Ensure reliability runs exist before the main consensus SC analysis.
subprocess.run([sys.executable,str(ROOT/'scripts/investigate_sc_reliability.py'),'--tracking-only'],check=True,cwd=ROOT)
from pat16.primary_sc import prepare
prepare()
from pat16.analysis import run
run()
cmd=[sys.executable,str(ROOT/'scripts/investigate_nuisance.py')]
subprocess.run(cmd,check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'scripts/primary_denoising_audit.py')],check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'scripts/investigate_censoring.py')],check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'scripts/investigate_edge_drivers.py')],check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'scripts/investigate_sc_reliability.py')],check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'scripts/recommend_primary_sc.py')],check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'scripts/investigate_joint_sampling.py')],check=True,cwd=ROOT)
for script in ['investigate_joint_refit.py','integrate_clinical_evidence.py','investigate_pc_sampling.py','recommend_primary_fc.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts'/script)],check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'scripts/build_notebooks.py')],check=True,cwd=ROOT)
if a.notebooks:subprocess.run([sys.executable,str(ROOT/'scripts/execute_notebooks.py')],check=True,cwd=ROOT)
if a.report:
    subprocess.run([sys.executable,str(ROOT/'scripts/report_tables.py')],check=True,cwd=ROOT)
    subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex'],check=True,cwd=ROOT/'report')
print(f'Completed in {time.time()-start:.1f}s')
