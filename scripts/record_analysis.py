"""Fingerprint the delivered scientific code and principal derived outputs."""
import hashlib,json
from pathlib import Path
from pat16 import ROOT,RESULTS

def record(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
paths=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+[ROOT/'README.md',ROOT/'pyproject.toml',ROOT/'.python-version',ROOT/'uv.lock',ROOT/'data/manifest.json',ROOT/'report/main.tex',ROOT/'report/references.bib']
paths+=list((ROOT/'config').glob('*'))+list((ROOT/'tests').glob('*.py'))
outputs=[RESULTS/name for name in ['summary.json','assessments_all.csv','graph_metrics.csv','network_summary.csv','sensitivity.csv','notebook_execution.json']]
for s in ['preop','postop']:
    outputs += [RESULTS/s/name for name in ['sc_counts.npy','sc_fraction.npy','fc_primary.npy','sc_qc.json','fc_qc.json','dwi_correction.json','dwi_corrected.nii.gz','dwi.bvec','dwi.bval','correction/acqparams.txt','correction/index.txt','correction/topup_command.json','correction/eddy_command.json','correction/field_Hz.nii.gz','correction/b0_unwarped.nii.gz','correction/topup_fieldcoef.nii.gz','correction/topup_movpar.txt','structural_fit_mask.nii.gz']]
    outputs += [RESULTS/'uncorrected_matched'/s/name for name in ['sc_counts.npy','sc_qc.json']]
    outputs += [RESULTS/s/name for name in ['bold_sdc.json','func_sdc.nii.gz','func_reference.nii.gz','func_motion.tsv','t1_to_func_pull.tfm','atlas_func.nii.gz','brain_mask_func.nii.gz','bold_valid_mask.nii.gz','bold_min_jacobian.nii.gz','bold_shift_vox.nii.gz','bold_sdc/field_BOLD_Hz.nii.gz','bold_sdc/bold_to_b0_pull.tfm','roi_timeseries_raw.npy','nuisance_pcs.npy','fc_primary_keep.npy']]
    outputs += list((RESULTS/s/'func_transforms').glob('*.tfm'))
    outputs += [RESULTS/'bold_uncorrected'/s/name for name in ['preserved.json','fc_primary.npy','fc_qc.json','roi_timeseries_raw.npy','nuisance_pcs.npy']]
    outputs += [RESULTS/'bold_readout_alternative'/s/name for name in ['func_sdc.nii.gz','fc_primary.npy','fc_qc.json','t1_to_func_pull.tfm','atlas_func.nii.gz','bold_valid_mask.nii.gz','bold_shift_vox.nii.gz','bold_min_jacobian.nii.gz','roi_timeseries_raw.npy','nuisance_pcs.npy']]
outputs += [RESULTS/name for name in ['correction_comparison.csv','dwi_correction_qc.csv','simulation_graph_growth.npy','simulation_lesion_sweep.csv']]
outputs += [RESULTS/name for name in ['bold_comparison.csv','bold_correction_qc.csv','bold_nuisance_sensitivity.csv','bold_comparison_common_nodes.npy','bold_validation.json','bold_uncorrected/README.txt','bold_uncorrected/report_before_BOLD_SDC.pdf']]
outputs += [ROOT/'report/main.pdf']+list((ROOT/'notebooks').glob('*.ipynb'))
outputs += list(RESULTS.glob('nuisance_*.csv'))
outputs += [RESULTS/'primary_pc_sensitivity.csv',RESULTS/'README_primary_denoising.txt']
outputs += [RESULTS/'cantab_all.csv',RESULTS/'psychosocial_all.csv']
outputs += [p for p in (RESULTS/'nuisance_investigation').rglob('*') if p.is_file() and p.suffix in ['.json','.npy','.gz','.pdf','.png']]
outputs += [RESULTS/'nuisance_validation.json',RESULTS/'primary_denoising_validation.json',RESULTS/'before_primary_anatomical5/manifest.json']
outputs += [p for p in (RESULTS/'censoring_investigation').rglob('*') if p.is_file() and p.suffix in ['.csv','.json','.npy','.npz','.pdf','.tex']]
outputs += [RESULTS/'censoring_validation.json']
outputs += [p for p in (RESULTS/'coupling_edge_drivers').rglob('*') if p.is_file() and p.suffix in ['.csv','.json','.npy','.npz','.pdf','.tex']]
outputs += [RESULTS/'edge_driver_validation.json']
outputs += [p for p in (RESULTS/'sc_reliability').rglob('*') if p.is_file() and p.suffix in ['.csv','.json','.npy','.npz','.pdf','.tex']]
outputs += [RESULTS/'sc_reliability_validation.json']
outputs += [p for p in (RESULTS/'primary_sc').rglob('*') if p.is_file()]
outputs += [RESULTS/'primary_sc_validation.json',RESULTS/'before_primary_sc/manifest.json']
outputs += [p for p in (RESULTS/'joint_sampling').glob('*') if p.is_file()]
outputs += [RESULTS/'joint_sampling_validation.json',RESULTS/'before_joint_sampling/manifest.json']
outputs += [p for folder in ['joint_refit','evidence_integration'] for p in (RESULTS/folder).glob('*') if p.is_file()]
outputs += [RESULTS/'joint_refit_validation.json',RESULTS/'before_joint_refit/manifest.json']
outputs += [p for p in (RESULTS/'pc_sampling').glob('*') if p.is_file()]
outputs += [RESULTS/'pc_sampling_validation.json',RESULTS/'before_pc_sampling/manifest.json']
outputs += [p for p in (RESULTS/'primary_fc').glob('*') if p.is_file()]
outputs += list((RESULTS/'primary_fc_audit').glob('*'))
outputs += [RESULTS/'primary_fc_validation.json',RESULTS/'before_primary_fc/manifest.json']
paths += [ROOT/p for p in ['.gitignore','.gitattributes','Makefile','THIRD_PARTY.md','CITATION.cff','LICENSE','LICENSE-CONTENT.md','.github/workflows/ci.yml']]
paths += list((ROOT/'docs').glob('*.md'))
for session in ['preop','postop']:
    for base in [RESULTS/session,RESULTS/'bold_readout_alternative'/session]:
        outputs += list(base.glob('primary_*.npy'))+list(base.glob('fc_*.npy'))+[base/'roi_timeseries_clean.npy']
outputs += list((RESULTS/'figures').glob('*.png'))+list((ROOT/'report/generated').glob('*.tex'))
obj={'analysis_date':'2026-10-18','patient':'sub-PAT16','classification':'exploratory single case; no validated clinical/predictive claims','primary_FC_resampling':'fixed-duration-nonreplacement-refit-v1; config/PRIMARY_FC_RESAMPLING.md; observed points separate from subset stability ranges','primary_SC':'repeat80-mean-seed-rate-common-coupling-v1; config/PRIMARY_SC.md','primary_denoising':'retained-anatomical5-primary-v1; post hoc methodological recommendation; config/PRIMARY_DENOISING.md','code_and_environment':[record(p) for p in sorted(paths)],'derived_outputs':[record(p) for p in sorted(set(outputs))],'input_integrity':'Verify data/manifest.json using scripts/manifest.py --verify','notebooks':'Executed with project-local Python; live widgets initialize on Run All','random_seeds':{'registration':160,'tractography':160,'block_bootstrap':160,'censoring_matching':3160,'joint_SC_bootstrap':'4160/4161; FC stream seeds in config/JOINT_SAMPLING.md','SC_reliability_preop':'17000–17019','SC_reliability_postop':'18000–18019'}}
(RESULTS/'analysis_provenance.json').write_text(json.dumps(obj,indent=2));print('Recorded analysis fingerprints')
