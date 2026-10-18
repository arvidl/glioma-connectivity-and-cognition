"""Fixed-model, independent random-seed-location tractography replicates.

These repeats measure numerical/seeding variability conditional on two scans;
they do not model acquisition, registration, CSD-fit or anatomical uncertainty.
"""
import inspect,json,time
from importlib.metadata import version
import numpy as np
import nibabel as nib
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.data import default_sphere
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.utils import random_seeds_from_mask
from dipy.tracking.streamline import length
from . import ROOT,RESULTS,SESSIONS
from .bold import sha
from .connectivity import endpoints_to_matrix

OUT=RESULTS/'sc_reliability'
REPEATS=20
SEEDS=30000
VERSION='fixed-csd-seeding-repeats-v1'


def track(coeff,fa,brain,atlas,affine,random_seed,seeds_count=SEEDS):
    getter=DeterministicMaximumDirectionGetter.from_shcoeff(coeff,max_angle=35.,sphere=default_sphere,pmf_threshold=.1)
    seeds=random_seeds_from_mask(brain&(fa>.25),affine,seeds_count=seeds_count,seed_count_per_voxel=False,random_seed=random_seed)
    tracker=LocalTracking(getter,ThresholdStoppingCriterion(fa,.1),seeds,affine,step_size=.75,max_cross=1,return_all=True,random_seed=random_seed,maxlen=400)
    streamlines=[s for s in tracker if 20<=length(s)<=250]
    counts,pairs,dist=endpoints_to_matrix(streamlines,atlas,affine)
    lens=np.array([length(s) for s in streamlines])
    return counts,{'pairs':pairs,'distances_mm':dist,'lengths_mm':lens},seeds


def run_session(session):
    p=RESULTS/session;out=OUT/session;out.mkdir(parents=True,exist_ok=True)
    names=['dwi_corrected.nii.gz','dwi.bval','dwi.bvec','fa.nii.gz','md.nii.gz','structural_fit_mask.nii.gz','atlas_dwi.nii.gz','sc_counts.npy','sc_qc.json','csd_sh_coeff.npy']
    spec={'version':VERSION,'inputs':{n:sha(p/n) for n in names},'code_sha256':sha(ROOT/'src/pat16/tract_repeats.py'),
          'endpoint_code_sha256':sha(ROOT/'src/pat16/connectivity.py'),'packages':{k:version(k) for k in ['dipy','numpy','nibabel']},
          'repeats':REPEATS,'seeds_per_run':SEEDS,'independent_random_seeds':[17000+1000*SESSIONS.index(session)+i for i in range(REPEATS)],
          'baseline_replay_seed':160,'tracking':{'step_mm':.75,'max_angle_deg':35.,'pmf_threshold':.1,'fa_seed':.25,'fa_stop':.1,'min_length_mm':20,'max_length_mm':250,'endpoint_radius_mm':6.,'max_cross':1,'maxlen':400}}
    complete=out/'complete.json'
    if complete.exists() and json.loads(complete.read_text())==spec:
        print(session,'tractography repeat cache current',flush=True);return
    # Checkpoints are valid only under the complete model/algorithm/input spec.
    pending=out/'design.json'
    if pending.exists() and json.loads(pending.read_text())!=spec:
        raise RuntimeError('Tractography repeat design changed; preserve old run directory before recomputing')
    pending.write_text(json.dumps(spec,indent=2))
    img=nib.load(p/'dwi_corrected.nii.gz');affine=img.affine
    fa=nib.load(p/'fa.nii.gz').get_fdata(dtype=np.float32)
    md=nib.load(p/'md.nii.gz').get_fdata(dtype=np.float32)
    brain=nib.load(p/'structural_fit_mask.nii.gz').get_fdata()>.5
    atlas=nib.load(p/'atlas_dwi.nii.gz').get_fdata().astype(int)
    coeff_path=out/'csd_sh_coeff64.npy'
    if not coeff_path.exists():
        # The original tracker used float64 fit coefficients, while its cache was
        # saved as float32. Refit the SAME CSD response/model once to recover the
        # original precision; an exact seed-160 count replay guards the bridge.
        data=img.get_fdata(dtype=np.float32);bval=np.loadtxt(p/'dwi.bval');bvec=np.loadtxt(p/'dwi.bvec').T
        use=(bval<50)|(np.abs(bval-2800)<100)
        gtab=gradient_table(bval[use],bvecs=bvec[use],b0_threshold=50)
        b0=data[...,bval<50].mean(-1);wm=(fa>.7)&brain&(md>0)&(md<.002)
        qc=json.loads((p/'sc_qc.json').read_text())
        response=(np.asarray(qc['response_eigenvalues_mm2_s']),float(np.median(b0[wm])))
        coeff=ConstrainedSphericalDeconvModel(gtab,response,sh_order_max=6).fit(data[...,use],mask=brain).shm_coeff
        np.save(coeff_path,coeff)
        del data
    coeff=np.load(coeff_path)
    if not (out/'baseline_replay.json').exists():
        t=time.time();counts,endpoints,seeds=track(coeff,fa,brain,atlas,affine,160)
        original=np.load(p/'sc_counts.npy')
        bridge={'seed':160,'seconds':time.time()-t,'exact_count_matrix_reproduction':bool(np.array_equal(counts,original)),
                'max_count_difference':int(np.abs(counts-original).max()),'length_filtered_streamlines':len(endpoints['pairs']),
                'original_length_filtered_streamlines':json.loads((p/'sc_qc.json').read_text())['length_filtered_streamlines'],
                'max_abs_coeff_difference_from_float32_cache':float(np.abs(coeff-np.load(p/'csd_sh_coeff.npy')).max())}
        np.save(out/'baseline_replay_counts.npy',counts)
        (out/'baseline_replay.json').write_text(json.dumps(bridge,indent=2))
        print(session,'baseline replay',bridge,flush=True)
    bridge=json.loads((out/'baseline_replay.json').read_text())
    if not bridge['exact_count_matrix_reproduction']:raise RuntimeError('Seed-160 baseline did not reproduce the original SC exactly')
    for i,seed in enumerate(spec['independent_random_seeds']):
        target=out/f'repeat_{i:02d}_counts.npy'
        if target.exists():continue
        t=time.time();counts,endpoints,seeds=track(coeff,fa,brain,atlas,affine,seed)
        np.savez_compressed(out/f'repeat_{i:02d}_endpoints.npz',**endpoints,seeds_ras_mm=seeds)
        info={'replicate':i,'random_seed':seed,'seeds':len(seeds),'length_filtered_streamlines':len(endpoints['pairs']),
              'assigned_interparcel_streamlines':int(counts.sum()//2),'nonzero_edges':int(np.count_nonzero(np.triu(counts,1))),
              'seconds':time.time()-t,'counts_sha256':sha(target) if target.exists() else None}
        np.save(target,counts);info['counts_sha256']=sha(target)
        (out/f'repeat_{i:02d}_qc.json').write_text(json.dumps(info,indent=2))
        print(session,'repeat',i+1,'/',REPEATS,'seed',seed,'edges',info['nonzero_edges'],'seconds',round(info['seconds'],2),flush=True)
    counts=np.stack([np.load(out/f'repeat_{i:02d}_counts.npy') for i in range(REPEATS)])
    np.save(out/'counts.npy',counts)
    complete.write_text(json.dumps(spec,indent=2))


def run():
    for session in SESSIONS:run_session(session)
