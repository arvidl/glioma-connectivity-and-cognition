"""Create didactic notebooks from literate source cells; no network at run time."""
import nbformat as n
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
COMMON='''from pathlib import Path
import os, sys, json
root = Path.cwd() if (Path.cwd()/"pyproject.toml").exists() else Path.cwd().parent
os.environ.setdefault("MPLCONFIGDIR", str(root/".cache/matplotlib"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, Markdown, Image
from pat16 import ROOT, DATA, RESULTS, SESSIONS
from pat16.connectivity import labels
from pat16.primary_sc import load_weights
nodes = pd.read_csv(RESULTS/"nodes.csv")
summary = json.loads((RESULTS/"summary.json").read_text())
plt.rcParams.update({"figure.figsize": (9,4), "font.size": 10})
'''

def M(s):return n.v4.new_markdown_cell(s)
def C(s):return n.v4.new_code_cell(s)
def write(name,title,cells):
    banner=M("**Primary FC stability:** notebook 14 recommends 97 distinct frames per session from nonreplacement ten-frame blocks, with complete WM/CSF nuisance refitting. These equal-duration subset ranges are conditional stability summaries, not full-run confidence intervals. The observed full-run points remain unchanged. Notebooks 10/11/13 retain earlier bootstrap, matching and mechanism analyses as sensitivities. See `config/PRIMARY_FC_RESAMPLING.md`. **Current SC policy:** graphs, network means and simulations use each session's 16/20 consensus with mean counts per 10,000 attempted seeds. Primary paired coupling uses the 823-edge intersection with session-specific weights. See `config/PRIMARY_SC.md`. Historical seed-160 diagnostics in notebooks 07/08 remain sensitivity analyses; notebook 09 supplies the reliability evidence. Notebook 10 quantifies joint seed/FC sampling under this policy, including matched designs and support reselection. Notebook 11 extends the full-run branch to paired nuisance refits; notebook 12 integrates all clinical/context fields with network uncertainty. Notebook 13 diagnoses PCA-training shifts and compares nonreplacement blocks at matched distinct counts and 80% retention. The rule is post hoc and does not validate anatomy or a surgical/cognitive effect.")
    b=n.v4.new_notebook(cells=[M(title),banner,C(COMMON)]+cells,metadata={'kernelspec':{'display_name':'Python 3 (uv project)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    n.write(b,ROOT/'notebooks'/name)

write('00_data_anatomy_and_design.ipynb','# 00 | From local tumour to distributed brain networks\n\n**Question:** What did we measure, and what can a paired single case tell us?\n\nRun with the project uv environment. All data, atlas files and computed results are bundled. Real data never fall back to random arrays. Orange/yellow lesion overlays refer to the **preoperative mask**, including when transported to postoperative space.',[
M('''## The evidence chain
T1 anatomy locates the lesion and provides a coordinate system. Diffusion MRI estimates local water-diffusion orientation; tractography builds an anatomical **estimate** of SC. Resting BOLD correlations estimate FC. CANTAB and questionnaires measure selected aspects of experience and performance. None of these is a direct measure of all cognition or quality of life.

The local release has pre-op reference SC/FC, but no paired supplied connectomes or full preprocessing derivatives. Our paired matrices are newly computed using one pipeline. The original reference uses DK68 and must **not** be subtracted from Schaefer100.

The patient is a 39-year-old male with the source diagnosis “Anaplastic astrocytoma II–III.” This historical description is preserved; no modern molecular diagnosis is inferred.'''),
C('''import nibabel as nib
rows=[]
for session in SESSIONS:
    for modality,pattern in [("T1","anat/*T1w.nii.gz"),("DWI AP","dwi/*AP_dwi.nii.gz"),("DWI PA","dwi/*PA_dwi.nii.gz"),("BOLD","func/*bold.nii.gz")]:
        p=next((DATA/f"sub-PAT16/ses-{session}").glob(pattern));im=nib.load(p)
        meta=json.loads(Path(str(p).replace('.nii.gz','.json')).read_text())
        row=dict(session=session,modality=modality,shape=im.shape,zooms=im.header.get_zooms(),orientation=nib.aff2axcodes(im.affine),PE=meta.get('PhaseEncodingDirection'),readout_seconds=meta.get('TotalReadoutTime'))
        if modality.startswith('DWI'):
            bv=np.loadtxt(str(p).replace('.nii.gz','.bval'));values,counts=np.unique(bv,return_counts=True)
            row['bvalue_counts_s_mm2']=dict(zip(values.astype(int),counts))
        rows.append(row)
display(pd.DataFrame(rows))
display(pd.read_csv(DATA/"sub-PAT16/sub-PAT16_sessions.tsv",sep="\\t"))'''),
M('''## Coordinate transforms are data
NIfTI voxel indices are not anatomical coordinates. T1 and EPI grids differ in orientation, spacing and field of view. We use SimpleITK fixed-to-moving pull maps in LPS millimetres; plotting uses Nibabel RAS coordinates. A native T1 lesion mask is resampled by its header, not copied by voxel index.

MNI→pre-op atlas projection uses a head-image affine fit followed by brain-only refinement. Post-op labels retain node identity via rigid longitudinal alignment outside the pre-op lesion. Rigid registration does not reconstruct resection-related tissue deformation. The MNI2009 rsHRF maps are **not** combined with the FSL MNI atlas.'''),
C('''display(Image(filename=str(RESULTS/"figures/registration_qc.png")))
from pat16.views import show_slice
widgets.interact(show_slice,session=widgets.Dropdown(options=SESSIONS),z=widgets.IntSlider(min=40,max=220,value=140),overlay=widgets.Dropdown(options=["atlas","dwi","func","preop lesion footprint"]))'''),
C('''display(nodes.sort_values("lesion_overlap_fraction",ascending=False)[["label","network","hemisphere","lesion_overlap_fraction","distance_to_lesion_mm"]].head(12))
print("Mask volume (ml):",round(summary["lesion_volume_ml"],2),"; source metadata:",summary["reported_tumor_volume_ml"])
print("Common FC nodes:",summary["common_fc_nodes"])
print(json.dumps(json.loads((RESULTS/"anatomy/complete.json").read_text()),indent=2))'''),
M('''## Exercise: quality before interpretation
Move through inferior, middle and superior slices for each modality. Identify mismatch at cortex, ventricles, lesion and skull base. Are the findings robust to discarding low-coverage parcels? The saved transforms permit external inspection.

**Boundary of evidence:** DWI uses FSL topup susceptibility correction and eddy-current/motion correction with rotated gradients and outlier replacement. BOLD now uses a rigidly transferred static AP/PA field and its own readout time. Slice timing remains uncorrected; unknown shim compatibility, field drift and readout metadata are limitations. No cavity segmentation, postoperative interval, medications, radiotherapy history, normative CANTAB scores or dedicated QoL scale is supplied.

**Provenance:** `uv run python scripts/manifest.py --verify` checks every copied input. Use `uv run python scripts/run_all.py --from-raw --force` to recompute.''')])

write('01_diffusion_structural_connectivity.ipynb','# 01 | Diffusion MRI → streamlines → structural graph\n\n**Learning goals:** distinguish diffusion tensors, crossing-fibre estimates, streamline endpoints and graph edges; explore construction sensitivity.',[
M('''## Diffusion is a measurement; streamlines are an inference
The signal model is $S(b,\\mathbf g)\\simeq S_0\\exp(-b\\,\\mathbf g^TD\\mathbf g)$. Tensor eigenvalues summarize anisotropy and mean diffusivity, but one tensor cannot resolve crossing fibres.

The actual AP protocol contains 6 b0, 16 b=700, 30 b=1200 and 50 b=2800 volumes (s/mm²). Weighted-least-squares DTI uses b0 + the 700/1200 shells for FA/MD. Single-shell, single-tissue CSD (spherical harmonic order 6) uses b0 + the 2800 shell only. Its response tensor is estimated on that same shell in measured high-FA voxels (low-shell FA>0.7). Deterministic tracking starts at 30,000 seeded locations (fixed RNG seed 160), uses a 0.75-mm step, 35-degree turning limit and FA stopping threshold 0.1. Keep 20–250-mm streamlines.

Raw DWI enters FSL **topup + eddy_cpu** without prior rigid resampling. Topup uses six AP b0 + two PA b0 volumes, AP `j-`, PA `j`, total readout **0.0266003 s**. Eddy corrects susceptibility, quadratic eddy currents and volume motion, and replaces flagged slices. Its **eddy_rotated_bvecs** feed reconstruction directly; no second rotation is applied. Native DWI is LAS and uses FSL voxel-axis b-vectors. Eight CPU threads suit Apple Silicon. No slice-to-volume or susceptibility-by-motion model, ACT/SIFT2, axonal-count inference or white-matter histology is implied.

**Revision of the earlier delivery:** the initial report incorrectly called the acquisition single-shell and fit CSD across shells. That analysis is archived under `results/legacy_baseline`. A new rigid-only comparator uses the SAME revised shell selection and tracking as the topup/eddy analysis, under `results/uncorrected_matched`. Compare the latter to assess preprocessing sensitivity; neither difference establishes biological accuracy.'''),
M('''## Audit the correction before trusting edges
The acquisition rows are generated from BIDS direction and readout metadata. Eddy's 102 index entries are all 1, pointing to the first AP b0. All eight topup volumes have corresponding acquisition rows. The supplied topup configuration, exact commands, logs, Hz field, Jacobians, movement RMS, outlier maps and corrected b0s are retained. Nonpositive AP or PA Jacobians are counted in QC and excluded from structural fitting.

The AP/PA voxel arrays are stacked on the AP grid without pre-interpolation, following fslmerge convention. The post-op prescriptions differ slightly (~0.8°); topup's rigid model approximates this difference. Inspect the field at skull base and lesion margins. An improved AP/PA fit is internal consistency, not independent anatomical validation.'''),
M('''### Metadata consistency exercise
The supplied readout is 0.0266003 s, whereas echo spacing × (96 phase lines − 1) gives 0.03610038 s. The scanner protocol is unavailable to resolve this discrepancy. We keep the supplied identical AP/PA value and pass the SAME acquisition table to topup and eddy. Shared scaling largely cancels in the displacement correction ([FSL FAQ](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/FAQ/index.html)); the exported physical Hz scale is conditional. The BOLD revision explicitly registers this field, uses BOLD readout timing and reports an alternative timing-ratio sensitivity; it is not an independently calibrated field. Read `readout_metadata_audit` in each session's correction JSON. Source sidecars remain unchanged.'''),
C('''display(pd.read_csv(RESULTS/"dwi_correction_qc.csv"))
from pat16.views import show_diffusion_correction
widgets.interact(show_diffusion_correction,session=widgets.Dropdown(options=SESSIONS),z=widgets.IntSlider(min=5,max=54,value=28))
display(Image(filename=str(RESULTS/"figures/dwi_motion_outliers.png")))'''),
C('''for session in SESSIONS:
    display(pd.Series(json.loads((RESULTS/session/"sc_qc.json").read_text()),name=session))
from pat16.views import show_tracts
widgets.interact(show_tracts,session=widgets.Dropdown(options=SESSIONS),n=widgets.IntSlider(min=50,max=1000,step=50,value=300))'''),
M('''## Define the edge explicitly
Each endpoint is assigned to the nearest labelled cortical voxel within 6 mm. Both endpoints must pass; within-parcel streamlines are discarded. $C_{ij}=C_{ji}$ counts assigned streamlines. Historical seed-160 fractions are retained as a sensitivity. Primary structural support requires nonzero counts in at least 16 of 20 new runs; primary weights are the mean count over ALL 20 runs (including zeros), divided by 30,000 attempted seeds and multiplied by 10,000. There is no additional count cutoff or renormalization to the retained total. Paired coupling uses the intersection of the two consensus supports, retaining each session's own weights.

A zero means **no streamline assigned by this pipeline**, not proven anatomical absence. Historical total-assignment fractions are compositional. The primary fixed attempted-seed denominator avoids that constraint, but seeding volume, geometry and yield remain sources of bias; it does not make weights axon counts. Node volumes, tracking yield and endpoint radius can all affect the result. Notebook [09](09_structural_edge_reliability.ipynb) generates repeated tractography and inspects low-count/session-specific edges, count thresholds and consensus coupling.'''),
C('''from pat16.views import connectome3d, show_plotly
widgets.interact(lambda session,density:show_plotly(connectome3d(session,"SC",density)),session=widgets.Dropdown(options=SESSIONS),density=widgets.FloatSlider(min=.01,max=.15,step=.01,value=.03))'''),
C('''sensitivity=pd.read_csv(RESULTS/"sensitivity.csv")
display(sensitivity[sensitivity.analysis.str.startswith("SC_")].dropna(axis=1,how="all"))
def endpoint_radius(radius=6):
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,session in zip(axs,SESSIONS):
        path=RESULTS/session/("sc_counts.npy" if radius==6 else f"sc_radius_{radius}mm.npy")
        c=np.load(path);im=ax.imshow(np.log1p(c),cmap="magma");ax.set_title(f"{session}: {c.sum()/2:.0f} assigned");fig.colorbar(im,ax=ax)
    plt.show()
widgets.interact(endpoint_radius,radius=widgets.SelectionSlider(options=[2,4,6,8],value=6))'''),
M('''## Processing sensitivity: matched reconstruction
Both variants use the same low-shell tensor, high-shell CSD, 30,000 seeds, node identities and endpoint rule. They differ in preprocessing and its downstream registration. The weighted comparison uses a single scale across all four graphs, so changes are not hidden by independent normalization. This is a pipeline sensitivity study, not a ground-truth tract validation.'''),
C('''display(pd.read_csv(RESULTS/"correction_comparison.csv"))
display(Image(filename=str(RESULTS/"figures/correction_comparison.png")))'''),
M('''## Exercise: an edge can change without an axon changing
Find parcels with many fewer assigned endpoints at 2 mm than 8 mm. Consider why cortical warping and resection affect endpoint matching. Inspect FA and tractogram rather than trusting edge-count changes alone.

The executable implementation is `src/pat16/connectivity.py:structural`. Re-running it reuses its completion checkpoint; the full pipeline's `--force` option invalidates checkpoints. The reference DK68 connectome is available under `data/bids/derivatives/TVB` for a separate method comparison, not a longitudinal surrogate.''')])

write('02_bold_functional_connectivity.ipynb','# 02 | BOLD time series → functional connectivity\n\n**Learning goals:** inspect motion, censoring, nuisance regression, degrees of freedom, signed correlations and analysis sensitivity.',[
M('''## FC is a statistical association
For parcel signals $x_i(t)$, $r_{ij}=\\mathrm{corr}(x_i,x_j)$. A correlation is neither a tract nor a causal connection. Tumour-related neurovascular uncoupling can alter BOLD even when neural signalling is preserved.

BOLD now receives susceptibility correction from the same-session DWI AP/PA topup field. A rigid, mutual-information fit transfers undistorted b0/Hz data into the BOLD reference, with two fixed refinement rounds after the initial fit. BOLD is `j-` with readout 0.01464 s. A signed voxel shift is $s=\\mathrm{PEsign}\\times f_{Hz}\\times T_{ro}$. Saved rigid motion maps output coordinates into each raw frame; the PE shift is then applied along that raw frame's voxel axis. Cubic sampling occurs once, with the Jacobian determinant of the composed pull.

The field is a static cross-acquisition approximation: DWI starts about 9–10 minutes after BOLD, no shim settings or BOLD-specific reversed-PE scan are supplied, and GE-EPI signal dropout is not recovered. Readout metadata are inconsistent. The primary uses the supplied BOLD/DWI readout ratio; an alternative uses echo spacing × (phase lines − 1) for both acquisitions, giving **0.761 times** the primary displacement. Registration is held fixed for this amplitude sensitivity. It does not cover all uncertainty in the field fit.

All 180 frames are spatially processed using the existing motion estimates. Positive Jacobians, field support and in-bounds sampling across all frames define a valid mask. The corrected mean is registered to T1 again, and atlas sampling, tissue-PC nuisance estimates and FC are recomputed. The prior rigid-only BOLD/FC is preserved under `results/bold_uncorrected`; SC remains unchanged. Slice timing and motion-dependent changes in the field remain uncorrected.

We realign all 180 frames to an early-run mean, drop the first 4 frames and censor approximate framewise displacement (FD)>0.5 mm plus the following frame. FD sums translation changes and absolute Euler-angle changes at a 50-mm head radius.

ROI means use cortical voxels with tSNR>5 outside the pre-op lesion footprint in both sessions; require ≥8 usable voxels. Primary nuisance regression uses 24 motion terms, five PCs from combined session-specific T1 WM/CSF cores (PVE≥0.95, native erosion and valid BOLD support), a linear trend and DCT terms below 0.01 Hz. Temporal projection, voxel scaling and PCA are trained only on retained original indices. Joint final regression preserves these indices. This custom CompCor-inspired recipe is a post hoc methodological recommendation; it is not independently validated tissue segmentation or an optimal patient-specific model. See `config/PRIMARY_DENOISING.md` and notebook 06 for rationale and sensitivities. Notebook [07](07_censoring_motion_duration_and_coupling.ipynb) investigates why strict censoring changes coupling, using matched duration/FD, run-half and nuisance-refitting controls. Primary FC has **no low-pass filter and no global signal regression**. A 0.01–0.1-Hz band-pass is a sensitivity analysis with an explicit residual-df guard.'''),
C('''display(pd.read_csv(RESULTS/"bold_correction_qc.csv"))
from pat16.views import show_bold_correction
widgets.interact(show_bold_correction,session=widgets.Dropdown(options=SESSIONS),z=widgets.IntSlider(min=5,max=36,value=22))
display(Image(filename=str(RESULTS/"figures/bold_distortion_qc.png")))'''),
M('''## Separate spatial correction from nuisance-model sensitivity
Compare rigid-only BOLD, corrected BOLD and the alternative timing ratio below, all refitted with the SAME anatomical primary recipe. The original rigid-only FC archive is retained separately. All comparisons use the same intersected nodes, saved motion estimates, censoring rule and fixed SC. Spatial masks, interpolation and nuisance PCs are refitted, so their difference is a **processing-package sensitivity**, not the isolated effect or validation of an unwarping algorithm.

The separately labelled HISTORICAL noncortical component-swap diagnostic holds either ROI time series or its five PCs fixed. It does not use the current anatomical primary recipe. It reveals that the large postoperative mean-FC change is mainly associated with refitting noncortical nuisance PCs. These are not validated WM/CSF components. The swaps are diagnostic, not independently preferred physiological models. A direction or magnitude of FC change cannot be attributed to surgery or better neural recovery from this exercise.'''),
C('''display(pd.read_csv(RESULTS/"primary_sc/bold_comparison.csv"))
display(Image(filename=str(RESULTS/"figures/primary_sc_bold.png")))
display(pd.read_csv(RESULTS/"bold_nuisance_sensitivity.csv"))'''),
M('''### Follow the mechanism in notebook 06
The postoperative five-PC cutoff includes a strongly cortical-mean-associated direction after refitting. A cubic **zero-field** resampling control reproduces much of this sensitivity, particularly with the new mask; it is not an isolated susceptibility effect. Notebook [06](06_nuisance_refitting_and_anatomical_denoising.ipynb) holds corrected cortical signals, motion and censoring fixed, sweeps PC count, and compares session-specific T1 WM/CSF models. Historical noncortical regression is retained as a sensitivity; the main analyses use retained-frame anatomical five-PC regression. Lower FC or greater variance removal is not physiological validation.'''),
C('''fig,axs=plt.subplots(2,1,figsize=(10,5),sharex=True)
for ax,session in zip(axs,SESSIONS):
    fd=np.load(RESULTS/session/"fd.npy");keep=np.load(RESULTS/session/"fc_primary_keep.npy")
    ax.plot(np.arange(len(fd))*2.4,fd,lw=1);ax.scatter(np.where(~keep)[0]*2.4,fd[~keep],s=12,color="crimson");ax.axhline(.5,color="orange",ls="--");ax.set_ylabel(f"{session} FD (mm)")
axs[-1].set_xlabel("Acquisition time (s)");plt.tight_layout();plt.show()
for session in SESSIONS:display(pd.DataFrame(json.loads((RESULTS/session/"fc_qc.json").read_text())["analyses"]))'''),
C('''from pat16.denoising import primary_fit

def denoising(session="preop",fd_threshold=.5,low_pass="none",gsr=False):
    # Use the SAME whole-brain GSR signal and censor-specific PCA refit as saved analyses.
    gs=np.load(RESULTS/session/"primary_global_signal.npy") if gsr else None
    try:
        fc,res,keep,q,fd=primary_fit(session,fd_threshold,gs,low_pass=.1 if low_pass=="0.1 Hz" else None)
    except RuntimeError as e:
        print("QC guard:",e);return
    display(pd.Series(q));fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(fc,vmin=-1,vmax=1,cmap="RdBu_r");fig.colorbar(im,ax=ax,label="Pearson r");plt.show()
widgets.interact(denoising,session=widgets.Dropdown(options=SESSIONS),fd_threshold=widgets.SelectionSlider(options=[.3,.5,1.],value=.5),low_pass=widgets.Dropdown(options=["none","0.1 Hz"]),gsr=False)'''),
M('''## Sampling uncertainty is not a surgery-effect confidence interval
The main network stability summary now uses 200 nonreplacement 97-frame subsets from ten-frame original-time tiles, with complete WM/CSF nuisance training refitted (notebook 14). The earlier 200 moving-block bootstrap summaries with nuisance fits held fixed remain sensitivity evidence in the original CSV tables. This probes within-run sampling instability; it does not capture registration error, neurovascular changes, acquisition variability or practice effects.

**Exercise:** compare GSR, stricter FD and the band-pass guard. Do regional trends survive? Low residual degrees of freedom should cause abstention, not a lower QC threshold. The tables retain missing values rather than replacing absent nodes with zero.'''),
C('''display(pd.read_csv(RESULTS/"sensitivity.csv").query("analysis in ['primary','fd03','fd10','gsr']").dropna(axis=1,how="all"))
display(pd.read_csv(RESULTS/"preop/fc_roi_coverage.csv").merge(nodes[["id","label","lesion_overlap_fraction"]],on="id").sort_values("usable_fraction").head(15))''')])

write('03_longitudinal_graphs_and_networks.ipynb','# 03 | Local injury, distributed graphs and SC–FC coupling\n\n**Question:** how much do conclusions depend on graph construction?',[
C('''policy=json.loads((RESULTS/"primary_sc/policy.json").read_text())
display(pd.DataFrame(policy["sessions"]).T)
display(Image(filename=str(RESULTS/"figures/primary_sc_choice.png")))
sc_sensitivity=pd.read_csv(RESULTS/"primary_sc/coupling_sensitivity.csv")
def compare_sc(design="consensus80_k1",support="common"):
    display(sc_sensitivity[(sc_sensitivity.design==design)&(sc_sensitivity.support==support)])
    gs=pd.read_csv(RESULTS/"primary_sc/graph_sensitivity.csv")
    display(gs[(gs.design==design)&(gs.support==support)&(gs.target_density==.2)])
widgets.interact(compare_sc,design=widgets.Dropdown(options=list(dict.fromkeys(sc_sensitivity.design)),value="consensus80_k1"),support=widgets.Dropdown(options=["common","own"],value="common"))'''),
M('''## Why two related supports?
Own consensus describes changing reconstructed topology, preserving 337/340 session-specific consensus connections. Common consensus asks how SC/FC ranks change on the **same connections**; it excludes those connections and uses both scans to define the population. Neither estimates axonal loss. Do not compare an own-support graph with a common-support coupling value as if they described identical edges.

All-run mean counts include misses, avoiding a detected-only mean. Per-seed rates avoid a session-total compositional constraint, but seed-mask size and tracking yield still differ. Multiplication by a positive scalar preserves Spearman ranks; session-specific normalization can alter weighted graph comparisons. No ACT/SIFT2 or biological calibration is implied. Compare original fractions, unfiltered mean counts, higher count cutoffs and 50/95/100% consensus above. The original fraction normalization is defined on assigned interparcel streamlines, whereas primary rates use attempted seeds.'''),
M('''## Keep node identity fixed
There are 100 Schaefer cortical parcels and 7 atlas network labels (Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default). Group-atlas labels are anatomical hypotheses in this patient, not individually validated functional networks. FC comparisons use the intersection of valid nodes in both sessions. SC and FC use the same identity table; subcortical and cerebellar pathways are omitted.

For weighted graph paths, edge cost is $\\ell_{ij}=1/w_{ij}$ and efficiency is $E=[n(n-1)]^{-1}\\sum_{i\\ne j}1/d_{ij}$; disconnected pairs contribute zero. We threshold positive edges over requested densities 0.10–0.30 and save the **achieved** density. SC efficiency uses one shared maximum-weight scale across sessions. FC uses positive edges at fixed scale 1; signed matrices remain available separately.'''),
C('''display(Image(filename=str(RESULTS/"figures/connectomes.png")))
display(pd.read_csv(RESULTS/"graph_metrics.csv").query("target_density==0.2"))'''),
C('''from pat16.networks import metrics

def inspect_graph(modality="SC",density=.2):
    common=np.load(RESULTS/"common_fc_nodes.npy");rows=[]
    scs=[load_weights(session) for session in SESSIONS];scale=max(w.max() for w in scs)
    for session in SESSIONS:
        w=load_weights(session) if modality=="SC" else np.load(RESULTS/session/"fc_primary.npy")
        if modality=="FC":w=np.maximum(w[np.ix_(common,common)],0)
        rows.append(dict(session=session,**metrics(w,density,scale if modality=="SC" else 1)))
    display(pd.DataFrame(rows))
widgets.interact(inspect_graph,modality=widgets.Dropdown(options=["SC","FC"]),density=widgets.FloatSlider(min=.05,max=.4,step=.05,value=.2))'''),
C('''display(Image(filename=str(RESULTS/"figures/primary_fc_network_sessions.png")))
display(Image(filename=str(RESULTS/"figures/threshold_sensitivity.png")))
for session in SESSIONS: print(session,"descriptive SC–FC Spearman rho:",summary["sessions"][session]["sc_fc_spearman"])'''),
C('''# Same SC, nodes, corrected cortical signals and censoring; alternative nuisance models.
denoising_models=pd.read_csv(RESULTS/"primary_sc/denoising_comparison.csv")
display(denoising_models.query("BOLD_processing=='corrected'")[["session","model","mean_fc_r","sc_fc_spearman","weighted_global_efficiency","residual_df"]])
print("SC–FC coupling and FC graph efficiency remain model-dependent; see notebook 06.")'''),
M('''## SC–FC coupling is a property of edges
We compute Spearman correlation between log(1+SC) and FC over the same 823 common-consensus upper-triangle edges in the common finite-FC node set, with session-specific repeat-mean weights. Spatial distance and atlas effects can influence this statistic. Edges are dependent, so no naïve edgewise p-value is reported. Notebook [07](07_censoring_motion_duration_and_coupling.ipynb) shows that strict censoring, frame matching and refitting change the contrast, and that the SC edge support/reference also matters. Its direction remains unresolved; the primary fit stays fixed. Notebook [08](08_edges_networks_and_coupling_reversal.ipynb) separates shared/session-specific SC edge terms, network pairs and reranked deletion effects. Notebook [09](09_structural_edge_reliability.ipynb) shows that repeated seeding, count cutoffs and consensus support further change the coupling population; historical seed-160 results remain labelled sensitivities.

**Exercise:** contrast unthresholded matrices with thresholded graph efficiency. Can a graph appear more efficient because weak edges were removed or weights renormalized? How does a shared scale differ from independently scaling each session to its own strongest edge? Do not treat node/edge counts as independent biological observations.''')])

write('04_cognition_and_lived_experience.ipynb','# 04 | CANTAB, psychosocial context and the limits of brain–behaviour inference\n\n**Learning goals:** read every assessment, preserve mixed outcomes and formulate testable network hypotheses.',[
M('''## Describe before explaining
Raw score change is $\\Delta = x_{post}-x_{pre}$. A direction-aligned percentage flips signs for latency, error and moves so rightward nominally means favorable. It is **not** a normative z-score, reliable-change index, effect size or global cognition score. Variability (SD) measures receive no favorable direction.

The supplied battery contains MOT, RTI, RVP, SOC and SSP. Reaction time and movement time differ; accuracy is already at ceiling in RTI. “A motor deficit” or “significant improvement” is not supported solely by these values. No control/retest distribution or assessment interval is supplied.'''),
C('''cantab=pd.read_csv(RESULTS/"cantab_all.csv")
display(cantab[["measure","description","units","preop","postop","delta_post_minus_pre","percent_change"]])
display(Image(filename=str(RESULTS/"figures/cognition.png")))'''),
C('''psych=pd.read_csv(RESULTS/"psychosocial_all.csv")
display(psych[["measure","preop","postop","delta_post_minus_pre"]])
display(Image(filename=str(RESULTS/"figures/psychosocial.png")))'''),
M('''## Measure dictionary: protocol, units and interpretation
The expanded report Appendices A.1/A.2 and this selector describe every field. The source JSON description is kept separately from our explanatory annotation. RTI reaction time ends at button release; movement time begins there. The unqualified `rti_five_rt` / `rti_five_mt` aggregates are not relabelled as means. RVP A-prime is not percent correct; `soc_prob_min_moves` is a **problem count** despite its name. SDs have no assigned favourable direction.

Questionnaire versions, education-code mapping and lifestyle-index formulas are unavailable. Post-op-only “Same/More/Less” fields are relative categories; missing baseline values are structural, not zero. No CANTAB norm, diagnosis or health-related QoL score is inferred.'''),
C('''dictionary=pd.read_csv(RESULTS/"assessments_all.csv").set_index("measure")
def inspect_measure(measure="rti_simple_rt_mean"):
    row=dictionary.loc[measure]
    display(row[["description","definition","domain","units","source_units","preop","postop","delta_post_minus_pre","direction_text","caveat","source_basis"]].to_frame("Value / explanation"))
widgets.interact(inspect_measure,measure=widgets.Dropdown(options=dictionary.index.tolist(),value="rti_simple_rt_mean",layout=widgets.Layout(width="550px")))'''),
M('''## Hypothesis map, not reverse inference
Speeded responding recruits sensory, attentional and motor systems; planning and visuospatial span have distributed control/attention contributions. Network labels do not identify the mechanism of an individual score change. Equal-weighted “cognition predictions” are not trained on two observations.

Anxiety fell while worry and loneliness rose, and employment changed from employed to sick leave. These are relevant to lived experience, but there is no dedicated health-related QoL instrument. Lower reappraisal and higher suppression have no universal favorable direction. The source's additional ERQ “processing”/“expression” labels are preserved without asserting standard validated ERQ subscales.'''),
C('''from pat16.clinical import HYPOTHESES
hypotheses=pd.DataFrame(HYPOTHESES,columns=["test","construct","candidate_networks","interpretation"])
display(hypotheses)
net=pd.read_csv(RESULTS/"network_summary.csv")

def domain_context(test="RTI"):
    row=hypotheses.set_index("test").loc[test];display(row)
    related=row.candidate_networks.split("; ")
    display(net[net.network.isin(related)][["session","network","within_fc_fisher_z","within_sc_fraction"]] if "within_sc_fraction" in net else net[net.network.isin(related)].filter(regex="session|network|^within_"))
    primary_ranges=pd.read_csv(RESULTS/"primary_fc/network_uncertainty.csv").query("role=='primary'")
    display(primary_ranges[primary_ranges.network.isin(related)])
    display(cantab[cantab.measure.str.startswith(test.lower()+"_")][["measure","preop","postop","delta_post_minus_pre"]])
widgets.interact(domain_context,test=widgets.Dropdown(options=hypotheses.test.tolist()))'''),
M('''## Why not correlate cognition with efficiency here?
With two time points a nonconstant correlation is mechanically ±1; it is not validation. No patient-specific predictive model or causal surgery effect can be estimated from these observations. Preserved performance despite changed connectivity could reflect redundancy, compensation, practice, measurement error or vascular effects; this design cannot separate them.

**Next study:** preregister network/domain hypotheses, obtain postoperative timing, treatment and medication details, matched retest controls or repeated assessments, normative CANTAB scores/reliability, validated QoL, neurovascular measures and improved preprocessing. Then evaluate predictions out of sample.''')])

write('05_simulations_growth_surgery_and_dynamics.ipynb','# 05 | Teaching simulations: from Fisher–KPP in tissue to graph dynamics\n\n**All concentrations, injury severities and dynamical predictions in this notebook are simulated.** Anatomy and baseline SC are patient-derived. No tumour growth rate, resection extent or cognitive outcome is fitted.',[
M('''## A network analogue of reaction–diffusion
The previous project models cell concentration in 3D tissue. A graph analogue is $\\dot c=-D Lc+\\rho c(1-c)$, where $L$ is the weighted graph Laplacian. It illustrates network-mediated spread and logistical growth; it is not a validated model of glioma migration along these streamlines. Space, vascular supply and local infiltration are omitted; time is arbitrary.

A hypothetical lesion scales each edge by endpoint survival: $W'_{ij}=(1-s e_i)W_{ij}(1-s e_j)$, where $e_i=\\exp(-d_i/10\\,\\mathrm{mm})$ uses the minimum parcel-to-pre-op-lesion distance. This is a scenario rule, **not an observed postoperative cavity or a surgical plan**.'''),
C('''from pat16.networks import graph_growth,attenuate,ou_fc,metrics
baseline=load_weights("preop");exposure=np.exp(-nodes.distance_to_lesion_mm.to_numpy()/10)
seed=int(nodes.lesion_overlap_fraction.argmax())

def growth_view(diffusion=.5,rho=.3,time=100):
    h=graph_growth(baseline,seed,steps=100,dt=.2,diffusion=diffusion,rho=rho)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for name in dict.fromkeys(nodes.network):axs[0].plot(np.arange(101)*.2,h[:,nodes.network==name].mean(1),label=name)
    axs[0].axvline(time*.2,color="black",ls="--");axs[0].set_xlabel("Arbitrary simulation time");axs[0].set_ylabel("Simulated concentration");axs[0].legend(fontsize=7)
    pts=axs[1].scatter(nodes.x,nodes.y,c=h[time],cmap="magma",vmin=0,vmax=1,s=40);fig.colorbar(pts,ax=axs[1],label="Simulated concentration");axs[1].set_xlabel("R (mm)");axs[1].set_ylabel("A (mm)");plt.show()
widgets.interact(growth_view,diffusion=widgets.FloatSlider(min=0,max=1,step=.1,value=.5),rho=widgets.FloatSlider(min=0,max=.6,step=.05,value=.3),time=widgets.IntSlider(min=0,max=100,value=100))'''),
M('''## From structure to fluctuating activity
An Ornstein–Uhlenbeck (OU) network obeys $d x=A x\\,dt+dB$, with $A=-I+gW/\\lambda_{max}(W_{pre})$ and $0\\le g<1$. Stationary covariance solves $A\\Sigma+\\Sigma A^T+I=0$. Normalize covariance to obtain model FC.

This linear, symmetric model has no BOLD haemodynamics, conduction delays, inhibition, neural adaptation or cognition. Its nonnegative SC cannot explain the full signed empirical FC. The baseline spectral scale stays fixed across injury severities; otherwise independent renormalization could hide damage.'''),
C('''def injury_view(severity=.5,gain=.6):
    injured=attenuate(baseline,exposure,severity);scale=baseline.max();rho=max(np.linalg.eigvalsh(baseline).max(),1e-12)
    pred=ou_fc(injured,gain,scale=rho)
    ratio=metrics(injured,1,scale)["weighted_global_efficiency"]/metrics(baseline,1,scale)["weighted_global_efficiency"]
    print(f"Hypothetical efficiency / baseline = {ratio:.3f}; gain = {gain:.2f}")
    fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].imshow(injured-baseline,cmap="RdBu_r");axs[0].set_title("Hypothetical SC change")
    im=axs[1].imshow(pred-np.eye(100),vmin=0,vmax=.1,cmap="magma");axs[1].set_title("OU model FC (diagonal omitted)");fig.colorbar(im,ax=axs[1]);plt.show()
widgets.interact(injury_view,severity=widgets.FloatSlider(min=0,max=1,step=.05,value=.5),gain=widgets.FloatSlider(min=0,max=.95,step=.05,value=.6))'''),
C('''display(Image(filename=str(RESULTS/"figures/simulations.png")))
display(pd.read_csv(RESULTS/"simulation_lesion_sweep.csv").iloc[::5])'''),
M('''## Exercise: loss, preservation and compensation
Set gain to zero: FC becomes the identity regardless of SC. Set severity to zero: recover baseline. At full severity only exposed edges attenuate. Can two networks with similar global efficiency have different FC? Which predictions change if injury acts on edges traversing the lesion instead of endpoints?

A future model may compare pressure release, fibre injury and plasticity scenarios against independent observations. This notebook supplies mechanisms to question, not evidence that a chosen mechanism explains PAT16.''')])
write('06_nuisance_refitting_and_anatomical_denoising.ipynb','# 06 | Why did postoperative FC depend so strongly on refitting five PCs?\n\n**Learning goals:** diagnose a nuisance-model discontinuity, distinguish spatial resampling from field correction, and compare anatomical component regression without selecting a model from the cognitive outcome.',[
M('''## A controlled question
All PC-count and image-by-mask diagnostics use **the same corrected cortical ROI signals**, 24 motion terms, trend/<0.01-Hz DCT, FD≤0.5-mm censoring, 100 nodes and corrected SC. Only nuisance signals change. The two complete BOLD pipelines retain their own ROI signals for the separate old/new package comparison.

The historical five PCs used all 180 frames, voxel centering/scaling, and an eroded noncortical mask; this mask was not tissue segmentation. The reconstructed PC subspace matches the saved historical PCs. Signs and orthogonal basis rotations do not affect the regression projection. A PC sign flip cannot explain the FC change.'''),
C('''display(pd.DataFrame(json.loads((RESULTS/"nuisance_investigation/summary.json").read_text())["sessions"]))
display(Image(filename=str(RESULTS/"figures/nuisance_pc_mechanism.png")))
from pat16.nuisance_views import show_cutoff,show_masks,show_model
def cutoff(session="postop",components=5):
    fig=show_cutoff(session,components);plt.show();plt.close(fig)
widgets.interact(cutoff,session=widgets.Dropdown(options=SESSIONS,value="postop"),components=widgets.IntSlider(min=0,max=20,value=5))'''),
M('''### What the cutoff reveals
After identical base motion/DCT regression, the old postoperative five PCs remove 22.5% of the **unweighted cortical ROI mean's residual variance**; the refitted five remove 71.3%. Old PC6 and new PC5 have strong cortical-mean associations, but their partial correlation is only 0.331: this is **not a simple one-to-one PC swap**. With six old PCs, postoperative mean FC drops from 0.268 to 0.156; with ten PCs the old/refitted means are both near 0.10. This diagnoses the cutoff; it does not establish the optimal component count or neural ground truth.

Partial R² measures in-sample variance removed beyond the base regressors. Shared neural activity and non-neural fluctuations can both contribute. It is not the fraction of noise removed.'''),
C('''factorial=pd.read_csv(RESULTS/"nuisance_mask_data_factorial.csv")
display(factorial.pivot(index=["session","image_for_PCA"],columns="mask_for_PCA",values="mean_fc_r"))
display(pd.read_csv(RESULTS/"nuisance_sampling_audit.csv"))'''),
M('''### Interpolation, masks and susceptibility
The cubic zero-field control resamples raw BOLD with the **same saved motion and composed cubic sampler**, but zero PE shift and unit Jacobian. It is a diagnostic PCA image; it does not refit ROI registration. The linear sampler is checked against historical SimpleITK data inside the nuisance mask. With the new postoperative mask, mean FC is 0.254 using old linear data, 0.143 using cubic zero-field data and 0.133 using corrected data. Interpolation therefore accounts for much of this diagnostic difference; the remaining difference includes the field and its intensity modulation. No causal percentage is assigned to interacting stages.

The postoperative old/new mask Dice is 0.877 (11,054 versus 9,335 voxels); 684 old voxels fail the corrected field-support check. `old_field_valid` removes these; `intersection` fixes common grid membership. Mixed image/mask combinations are numerical controls, not anatomically preferred complete pipelines. Physical tissue correspondence changes after unwarping.'''),
M('''## Anatomical WM/CSF masks: inspect before interpreting
FSL FAST segments each session's native T1 into CSF/GM/WM partial-volume estimates with bias correction. The existing T1 brain mask excludes the lesion footprint plus 5 mm before fitting; the post-op footprint is not a cavity segmentation. Tissue fractions ≥0.95 are eroded by 1 mm in T1 space and restricted to >2 mm inside the brain. Native core membership (nearest neighbour) **and** interpolated PVE≥0.95 must pass in BOLD space, together with valid support and lesion exclusion. CSF includes ventricular and sulcal tissue. Masks are automatically estimated and need expert review.

Orange = WM; cyan = CSF; magenta = historical noncortical mask. A high PVE-derived purity is internal model consistency, not independent segmentation accuracy.'''),
C('''display(pd.read_csv(RESULTS/"nuisance_mask_composition.csv").query("BOLD_processing=='corrected'"))
def masks(session="postop",z=145):
    fig=show_masks(session,z);plt.show();plt.close(fig)
widgets.interact(masks,session=widgets.Dropdown(options=SESSIONS,value="postop"),z=widgets.IntSlider(min=40,max=220,value=145))'''),
M('''## Primary recommendation and five alternative nuisance models
**Primary: retained-frame anatomical five-PC regression.** Anatomical specificity and exclusion of censored samples from PCA motivate this choice; five components preserve the earlier budget (107/111 residual df). Tissue means are parsimonious but cannot represent multiple spatial noise directions; separate WM5+CSF5 uses ten PCs and remains a useful sensitivity. Combined PCA has more WM than CSF voxels, so tissue contributions are not balanced. No model is selected by cognitive agreement, lowest FC or highest coupling. This is a post hoc recommendation, not preregistration or proof of physiological superiority. General CompCor/aCompCor evidence supports tissue-based components, while cohort benchmarks show tradeoffs (Behzadi 2007; Muschelli 2014; Ciric 2017; see report).

All use joint final regression with the same motion/DCT terms and no explicit GSR. `anatomical5_raw` changes only the mask relative to historical all-frame PCA. `anatomical5` first projects constant/linear/<0.01-Hz DCT, scales voxels and fits PCA **on retained original indices**, then projects all frames with the fitted spatial weights. No temporal concatenation or interpolation across censor gaps occurs. `wm5_csf5` uses 5 PCs per tissue (10 total); `wm_csf_means` uses two tissue means. This is custom anatomical component regression inspired by CompCor, not a claim to reproduce fMRIPrep.

The anatomy-based five-PC model gives mean FC 0.087 pre-op and 0.046 post-op, versus 0.360/0.133 historically; coupling is evaluated on the primary 823-edge common consensus in the table below; original seed-160 coupling remains available in the archived CSV. It removes about 91%/90% of cortical-mean residual variance. This can remove both physiological artifacts and neural covariance. There are no physiological recordings or independently validated tissue labels to rank biological accuracy.'''),
C('''models=pd.read_csv(RESULTS/"primary_sc/denoising_comparison.csv")
display(models.query("BOLD_processing=='corrected'")[["session","model","nuisance_regressors","retained_frames","residual_df","mean_fc_r","sc_fc_spearman","cortical_mean_partial_R2","mean_ROI_partial_R2"]])
display(Image(filename=str(RESULTS/"figures/primary_sc_denoising.png")))
def matrix(session="postop",model="anatomical5"):
    fig=show_model(session,model);plt.show();plt.close(fig)
widgets.interact(matrix,session=widgets.Dropdown(options=SESSIONS,value="postop"),model=widgets.Dropdown(options=models.model.unique().tolist(),value="anatomical5"))'''),
C('''display(models.query("model in ['original5','anatomical5_raw','anatomical5']")[["session","BOLD_processing","model","mean_fc_r","sc_fc_spearman"]])
anatomical_sensitivity=pd.read_csv(RESULTS/"nuisance_anatomical_sensitivity.csv")
def sensitivity(session="postop",threshold=.95,components=5):
    display(anatomical_sensitivity[(anatomical_sensitivity.session==session)&(anatomical_sensitivity.PVE_threshold==threshold)&(anatomical_sensitivity.components==components)])
widgets.interact(sensitivity,session=widgets.Dropdown(options=SESSIONS,value="postop"),threshold=widgets.SelectionSlider(options=[.90,.95,.99],value=.95),components=widgets.SelectionSlider(options=[2,5,10],value=5))
display(pd.read_csv(RESULTS/"nuisance_network_summary.csv").query("BOLD_processing=='corrected' and model in ['original5','anatomical5']"))'''),
M('''## Audit the recommended five-PC cutoff
A 0–20 anatomical-PC sweep uses the same saved tissue PCA, fixed masks and retained frames. Adjacent postoperative 4/5/6-PC means are 0.0477/0.0456/0.0449, much closer than the historical cutoff jump; preoperative means are 0.0864/0.0872/0.0721. This check supports a pragmatic five-component budget, not an optimized count or neural validation. Wider count dependence remains visible.'''),
C('''primary_counts=pd.read_csv(RESULTS/"primary_sc/pc_sensitivity.csv")
display(Image(filename=str(RESULTS/"figures/primary_sc_pc_count.png")))
def adjacent_count(components=5):
    display(primary_counts[(primary_counts.BOLD_processing=="corrected")&(primary_counts.components==components)])
widgets.interact(adjacent_count,components=widgets.IntSlider(min=0,max=20,value=5))'''),
M('''## What survives, and what remains uncertain?
The anatomy-based five-PC old/new BOLD gap is small postoperatively (0.048 versus 0.046), compared with 0.272 versus 0.133 historically. At 0.90–0.99 tissue thresholds, corrected means range 0.084–0.089 pre-op and 0.045–0.048 post-op. Component count still matters. Stricter FD≤0.3-mm censoring refits anatomical PCs on its retained frames and leaves only 29/57 residual degrees of freedom; unequal, shorter runs limit the comparison. Under the current common-consensus SC rule, coupling is 0.188→0.146 at FD0.5 and 0.161→0.127 under strict FD. The earlier 0.210→0.189 versus 0.156→0.168 reversal used original own-support SC and remains a historical sensitivity. Notebook 07 separates frame restriction, coefficient/PCA refitting, matched duration/FD/run halves and SC-support dependence; its matched refits also leave the direction unresolved.

The historical primary FC/report/tables are frozen under `results/before_primary_anatomical5`; FC still uses `anatomical5`; primary SC now uses repeated-seed consensus as specified above. Original SC arrays remain immutable. The previous main delivery is archived under `results/before_primary_sc`. All six models, masks, regressors and diagnostics remain under `results/nuisance_investigation`. Regenerate using `uv run python scripts/investigate_nuisance.py`; FSL FAST is needed only if tissue segmentation is absent or deliberately forced. Report Appendix A explains the assessments used in notebook 04. The primary recipe is fixed across sessions and is not chosen for agreement with cognition; no two-time-point correlation is interpreted as evidence.

**Exercises:** compare five with six PCs; compare raw anatomical with retained-frame anatomical PCA; inspect tissue masks near the ventricles, cortex and lesion; ask which claims depend on mean FC versus SC–FC coupling. Which independent physiological or retest data would help distinguish artifact removal from neural signal loss?''')])
write('07_censoring_motion_duration_and_coupling.ipynb','# 07 | Why does stricter censoring reverse SC–FC coupling?\n\n**Learning goals:** separate frame selection from nuisance refitting; match retained exposure and motion; distinguish SC support from FC changes; interpret conditional resampling without claiming a treatment effect.',[
C('''display(pd.read_csv(RESULTS/"primary_sc/censoring_decomposition.csv"))
display(pd.read_csv(RESULTS/"primary_sc/matched_coupling_summary.csv"))
print("Current common-consensus coupling has negative primary and strict point contrasts. The original reversal investigation below is retained as a historical sensitivity.")'''),
M('''## Define the contrast before explaining it
In the historical seed-160 own-support experiment, the anatomical5 pipeline uses FD≤0.5 mm plus following-frame censoring, 146/150 frames and 107/111 algebraic residual df. Coupling is 0.2101→0.1894 (post−pre **−0.0207**). Strict FD≤0.3 mm retains 68/96 frames, 29/57 df, and refits tissue PCA: 0.1564→0.1675 (**+0.0111**). Both use the same corrected images, cortical ROIs, motion and session SC. A smaller mean FC is not equivalent to lower SC–FC rank coupling.

The historical experiments below used original seed-160 SC and left the then-primary matrices unchanged. Their coupling uses each session's own nonzero SC edges (1515/1559), with no graph-density threshold. The SC-support controls below explicitly change that estimand. No model is chosen from cognition or a preferred sign.'''),
C('''from pat16.censoring import OUT, PROTOCOLS, LABELS, show_draw
spec=json.loads((OUT/"design.json").read_text())
frame_qc=pd.read_csv(OUT/"frame_qc.csv")
display(frame_qc)
display(Image(filename=str(RESULTS/"figures/censoring_frame_qc.png")))'''),
M('''### Shorter exposure and different temporal coverage
Strict retained-frame mean FD is 0.2170/0.2088 mm (pre/post), versus 0.2869/0.2522 primarily. These are **retained-frame** summaries; the original main QC reports mean FD over all frames after the initial four. Strict censoring retains 35/12/21 pre-op frames in acquisition thirds, versus 39/34/23 post-op; 46/68 pre-op frames lie in the first half. The remaining samples span 6.96/6.76 minutes, but provide only 2.72/3.84 minutes of retained exposure. Retained exposure, acquisition span, run fragmentation and independent sample size are different quantities. Scan length and acquisition span both influence FC reliability (Birn et al., 2013; report bibliography).

## An ordered computational decomposition
First restrict **already denoised primary residuals** to strict frames; next refit final regression coefficients while keeping primary PCs fixed; finally refit tissue temporal projection/scaling/PCA on strict frames. These increments are order dependent. Frozen-fit controls use training observations outside the selected subset and are diagnostic, not independent short-run pipelines.'''),
C('''decomposition=pd.read_csv(OUT/"decomposition.csv")
display(decomposition[["session","stage","frames","residual_df","df_meaning","coupling","common_coupling","mean_fc"]])
delta=decomposition.pivot(index="stage",columns="session",values="coupling").loc[["primary","restriction","coefficients","pca"]]
delta["post_minus_pre"]=delta.postop-delta.preop
display(delta)
display(pd.DataFrame(json.loads((OUT/"pc_subspaces.json").read_text())))
display(Image(filename=str(RESULTS/"figures/censoring_decomposition.png")))'''),
M('''The contrast moves **−0.0207 → −0.0116 → −0.0027 → +0.0111**. Frame restriction alone does not reverse it. Refitting regression reduces coupling more preoperatively; retraining the five-PC subspace then crosses zero. Mean squared canonical correlations between primary/strict PC score subspaces centered on strict frames are 0.635 pre-op and 0.740 post-op. Signs and orthogonal basis rotations cannot cause a projection change. The strict pre-op fit has only 29 algebraic residual df; these are not autocorrelation-adjusted effective df. This sequence identifies computational dependencies, not noise versus neural truth.

## Matched designs: six protocols × 200 draws × two fitting modes
Every draw retains **68 frames/session**, without replacement, on the original 180-frame axis. Complete refits retrain temporal nuisance preprocessing, scaling, PCA and final regression using only selected frames (39 rank, 29 residual df). Frozen mode holds **both primary PCs and final regression coefficients** fixed; only the correlation samples change.

For each primary/strict pool, compare duration-only uniform sampling, duration+FD matching, and duration+FD+run-half matching. FD matching uses randomized maximum-cardinality bipartite assignment, a **0.015-mm absolute pairwise caliper**, then selection of 68 pairs. Half matching also gives equal counts within frames 0–89 and 90–179. No FC/SC/cognition determines selection. This targets common FD support, not uniform samples of the original pool, and matched frame pairs do not imply matching neural states.

Why this caliper? At 0.010 mm the strict pools permit 68 FD-only pairs but only 65 half-matched and 56 third-matched pairs; at 0.015 mm half matching permits 68, thirds only 60. We use 0.015 consistently for the motion controls. Half matching is a coarse temporal check, not exact time/state matching. The strict pre-op pool is exactly 68 frames and is unchanged across strict draws. This asymmetric conditional design is not participant resampling.'''),
C('''matched=pd.read_csv(OUT/"matched_replicates.csv")
matched_summary=pd.read_csv(OUT/"matched_summary.csv")
display(matched_summary[["protocol","mode","coupling_delta_median","coupling_delta_q025","coupling_delta_q975","coupling_positive_fraction","fd_abs_smd_median","fd_abs_smd_max_abs","fd_wasserstein_mm_median","first_half_count_difference_median"]])
display(Image(filename=str(RESULTS/"figures/censoring_matching.png")))'''),
M('''### What matching resolves—and what it does not
For full refits, primary-pool median changes remain negative (−0.0199 duration, −0.0215 duration+FD, −0.0240 with half matching), but every central range includes zero. Strict-pool medians are **−0.0069**, **−0.0135**, and **+0.0101**, respectively, with central ranges also spanning zero. FD-matched median absolute standardized differences are about 0.013–0.020 (all draws <0.070), much smaller than duration-only values. Balance on FD does not establish physiological equivalence or artifact removal.

The half control changes strict post-op first-half occupancy from a median 40 to exactly 46 frames, equal to pre-op. Its positive median with refitting, versus a negative frozen-fit median, shows that coarse time selection and nuisance refitting interact even after FD and frame counts are matched. The median third counts still differ: half matching cannot equate finer temporal coverage, fragmentation, autocorrelation or vigilance. The results therefore **do not isolate unequal duration as the sole explanation**, and do not establish neural recovery or decline.

The empirical 2.5/97.5 percentiles describe **conditional sampling-design spread**, not confidence intervals, probabilities of a surgery effect, p values or test–retest reliability. Single-frame draws do not model serial dependence. A high fraction of positive draws is descriptive only.'''),
C('''def inspect_draw(protocol="strict_motion_time",mode="refit",replicate=0):
    fig=show_draw(protocol,mode,replicate);plt.show();plt.close(fig)
widgets.interact(inspect_draw,protocol=widgets.Dropdown(options=list(PROTOCOLS),value="strict_motion_time"),mode=widgets.Dropdown(options=["frozen","refit"],value="refit"),replicate=widgets.IntSlider(min=0,max=spec["repeats"]-1,value=0,continuous_update=False))'''),
M('''## Hold SC support fixed
The two own-SC supports have **1071 common nonzero edges**. Restricting to this intersection while retaining each session's weights yields primary coupling 0.2031→0.1630 (−0.0402) and strict coupling 0.1658→0.1282 (−0.0376): no reversal. Keeping **pre-op SC weights** fixed on that intersection instead yields +0.0064 primarily and +0.0019 strictly. These are distinct questions: own SC compares the two full reconstructed SC–FC associations; fixed pre-SC asks how FC aligns with a single structural reference. Neither is intrinsically the correct biological estimand.

Explore all three support/reference definitions on the same matched draws. The own-support reversal is not a universal property of FC, and the across-session SC reconstruction contributes to the measured contrast. Support controls do not establish whether changed SC edges are true anatomy or tractography variability.'''),
C('''display(decomposition.query("stage in ['primary','pca']")[["session","stage","coupling","common_coupling","fixed_pre_sc_coupling"]])
def support_view(reference="common_coupling",mode="refit"):
    cols=["protocol","mode",reference+"_delta_median",reference+"_delta_q025",reference+"_delta_q975",reference+"_positive_fraction"]
    display(matched_summary[matched_summary["mode"]==mode][cols])
    fig,ax=plt.subplots(figsize=(9,3.5))
    for p in ["strict_duration","strict_motion","strict_motion_time"]:
        g=matched[(matched.protocol==p)&(matched["mode"]==mode)]
        w=g.pivot(index="replicate",columns="session",values=reference)
        ax.hist(w.postop-w.preop,bins=22,histtype="step",label=p)
    ax.axvline(0,color="black",lw=.8);ax.set(xlabel="Post − pre coupling",ylabel="Conditional draws");ax.legend(fontsize=8);plt.show();plt.close(fig)
widgets.interact(support_view,reference=widgets.Dropdown(options=["coupling","common_coupling","fixed_pre_sc_coupling"],value="common_coupling"),mode=widgets.Dropdown(options=["frozen","refit"],value="refit"))'''),
M('''## Reproducibility and exercises
`uv run python scripts/investigate_censoring.py` reproduces 200 draws per protocol from bundled data, masks and SC, without rerunning FSL. `design.json` records source/code SHA256 and seeds; `matched_masks.npz` stores all selected masks in preop/postop order; CSVs contain every draw's balance, temporal coverage, mean FC and three coupling estimands. The code never overwrites primary or strict FC. See `config/CENSORING_MATCHING.md` for assumptions and feasibility. Earlier primary report/provenance and immutable-output hashes are under `before_matching`.

**Conclusion:** retain the existing FD0.5 anatomical5 primary strategy on methodological grounds, with the coupling direction explicitly unresolved across sensitivity designs. Do not select a censor threshold or SC reference from the desired sign. The result is compatible with interacting selection, small-sample regression/PCA and SC-support effects; no independent physiological or retest evidence identifies their biological components.

**Exercises:** compare frozen and fully refitted results for the same saved draw; compare motion-only versus half matching and inspect remaining third-count differences; change the SC reference without changing FC; explain why 68 frames with rank 39 do not supply 29 independent neural observations. Which additional BOLD runs, physiological recordings or repeat SC reconstructions would discriminate these explanations? Continue in notebook [08](08_edges_networks_and_coupling_reversal.ipynb) to inspect individual edge contributions, shared/session-specific supports and network-pair/deletion controls.''')])

write('08_edges_networks_and_coupling_reversal.ipynb','# 08 | Which edges and networks account for the coupling reversal?\n\n**Learning goals:** distinguish shared from session-specific SC supports; reconstruct Spearman coupling from edge rank products; compare additive terms with reranked deletion; inspect atlas edges and conditional matched controls.',[
C('''display(pd.read_csv(RESULTS/"primary_sc/network_contributions.csv"))
print("Current accounting sums to primary Δ=-0.04173, strict Δ=-0.03424, and their difference +0.00749. Historical accounting below concerns the original +0.03186 reversal shift.")'''),
M('''## Define what is being localized
Let Δ(a)=ρ(post,a)−ρ(pre,a). The primary Δ is −0.0207203 and strict Δ is +0.0111444. Here **R=Δ(strict)−Δ(primary)=+0.0318647** is the *sensitivity of a longitudinal statistic to censoring/refitting*, not a surgical effect. Notebook 07 investigates the sampling/refitting sequence; this notebook allocates that sensitivity to measured edge populations.

The SC union has **2003** undirected edges: **1071 shared, 444 pre-op-only and 488 post-op-only**. “Only” refers to this tractography reconstruction, not proven tract loss/gain. Edges absent from both supports are omitted. Coupling uses positive SC edges without graph-density thresholding. All inputs of this historical seed-160 analysis are preserved; current primary consensus results are shown above.

### An exact additive identity
Spearman coupling is Pearson correlation of average ranks (ties retain their average rank). For centered SC and FC ranks x,y in one session's **full nonzero-SC support**, define cₑ=xₑyₑ/√(Σx²Σy²). Then Σcₑ=ρ exactly. Counts and log1p(SC fractions) have the same ranks. On the union, an absent edge contributes zero *to accounting*; its FC is never zero-imputed or ranked.

Define rₑ=(c(post,strict)−c(pre,strict))−(c(post,primary)−c(pre,primary)). Sum rₑ by support or unordered network pair to recover R. These are global-rank bookkeeping terms, **not subgroup correlations, causal effects or independent-edge importance scores**. Ranks and normalizers change when the edge universe changes.'''),
C('''from pat16.edge_drivers import OUT, GROUPS, GROUP_LABELS, NETWORKS, show_edges
edges=pd.read_csv(OUT/"edges.csv")
support=pd.read_csv(OUT/"support_summary.csv").set_index("support_group").loc[GROUPS]
context=pd.read_csv(OUT/"rank_context.csv")
display(support[["edges","primary_delta_contribution","pca_delta_contribution","reversal_shift","preop_median_count","postop_median_count","preop_fraction_counts_le2","postop_fraction_counts_le2"]])
print("Sum of support R terms:",support.reversal_shift.sum())
display(Image(filename=str(RESULTS/"figures/edge_driver_support.png")))'''),
M('''## Shared and session-specific edges
Pre-op-only edges supply **+0.02437**, shared edges **+0.00833**, and post-op-only edges **−0.00084** to R. This is signed accounting with cancellation, not a fraction of biological variance explained. Pre-only/post-only median SC counts are 2; 61.9%/66.4% have only 1–2 streamlines, versus 29.1%/28.3% among shared edges. Low counts do not prove invalidity, but support labels depend on reconstruction and endpoint rules.

The pre-only pooled coupling contribution falls from +0.05540 to +0.03102 when strict denoising is used. Mean FC rises from 0.1046 to 0.1338 and mean **relative FC rank** rises from 0.4293 to 0.4543. These low-SC edges become less uniformly low-FC in the pooled ordering, shrinking the positive low-SC/low-FC association. Both within-group rank covariance and between-group rank-location terms change. This does not identify whether removed covariance was artifact or neural.

The shared contribution here uses ranks among *all* own-session SC edges. Coupling recomputed on shared edges alone is a different statistic: primary Δ=−0.0402 and strict Δ=−0.0376, as shown in notebook 07. Do not average subgroup Spearman coefficients to reconstruct the whole.'''),
C('''display(context[context.stage.isin(["primary","pca"])][["session","stage","support_group","edges","mean_fc","mean_fc_rank_fraction","pooled_contribution","within_group_global_rank_term","between_group_global_rank_term","subgroup_spearman"]])
stage_terms=edges.groupby("support_group")[["restriction_shift","coefficient_refit_shift","pca_refit_shift","reversal_shift"]].sum()
display(stage_terms)
print("Within/between identity error:",np.max(abs(context.pooled_contribution-context.within_group_global_rank_term-context.between_group_global_rank_term)))'''),
M('''## Network pairs: positive terms coexist with substantial offsets
The largest positive R terms are **Default–SomMot +0.01237**, **Default–SalVentAttn +0.00896**, and **Default–Limbic +0.00644**. Within-Vis and within-SomMot also contribute positively. The largest opposing terms include **within-Default −0.00700**, Vis–SomMot −0.00593, DorsAttn–SomMot −0.00492 and Cont–SomMot −0.00490. The network-pair heatmaps separate all three support groups using the same color scale.

Each undirected pair is shown symmetrically, but counted **once** in sums. Pairs partition the edge set; network-incident edge sets overlap. Atlas network labels are not independently validated functional networks in PAT16, and a prominent pair cannot be equated with a cognitive mechanism. Group size, SC ties, FC ranking and positive/negative cancellation influence these totals.'''),
C('''pairs=pd.read_csv(OUT/"network_pairs.csv")
display(pairs.sort_values("reversal_shift",ascending=False)[["pair","edges","shared_shift","pre_only_shift","post_only_shift","reversal_shift","shift_per_edge"]])
display(Image(filename=str(RESULTS/"figures/edge_driver_networks.png")))'''),
M('''### Inspect the individual edges
The largest positive term is edge **21–42**, LH_DorsAttn_Post_6 ↔ LH_Default_PFC_1 (pre-only, 1/0 streamlines), rₑ=+0.00277. The largest negative term is edge **16–35**, LH_DorsAttn_Post_1 ↔ LH_Cont_PFCl_1 (shared), rₑ=−0.00287. A pair can contain large terms of both signs even when its net term is small or opposite. Across all edges, positive terms sum +0.4165 and negative terms −0.3846: strong cancellation yields the modest net +0.0319.

Use source IDs/labels to inspect the full inventory. Straight lines below connect atlas centroids in pre-op T1 RAS coordinates; **they are not reconstructed tracts**. A near-footprint endpoint does not show that a tract crosses tumour or cavity. The endpoint≤10-mm group contributes −0.01463, versus +0.04649 when both endpoints are farther away; this sensitivity is not confined to lesion-near atlas endpoints.'''),
C('''def edge_browser(support="all",pair="all",top=10,direction="positive"):
    fig,chosen=show_edges(support,pair,top,direction);plt.show();plt.close(fig)
    display(chosen[["node_i","node_j","label_i","label_j","support_group","sc_pre","sc_post","reversal_shift","leave_one_primary_delta","leave_one_pca_delta"]])
widgets.interact(edge_browser,support=widgets.Dropdown(options=["all"]+GROUPS,value="all"),pair=widgets.Dropdown(options=["all"]+sorted(pairs.pair.tolist()),value="all"),top=widgets.IntSlider(min=1,max=30,value=10,continuous_update=False),direction=widgets.Dropdown(options=["positive","negative"],value="positive"))'''),
M('''## Deleting edges changes every remaining rank
All **2003** single-edge deletions preserve the original negative-primary/positive-strict contrast. No individual edge is necessary for the sign reversal. Removing the top ten positive rₑ edges jointly makes strict Δ slightly negative (−0.00109), but this set was chosen from these same observations: it is a descriptive stress test, not validation of ten causal pathways.

Deleting **pre-only** edges leaves primary/strict Δ=−0.0137/+0.0017; deleting **post-only** edges gives −0.0472/−0.0282. This differs from the additive accounting: the post-only direct R term is small, yet its removal changes the remaining rank context. Removing both session-specific groups gives the shared-only negative contrasts.

Removing **all Default-incident edges** makes both contrasts positive (+0.0367/+0.0393), while removing **SalVentAttn-, Limbic-, or Cont-incident edges** makes both negative. Thus no unique support class, network or edge can be labelled the biological cause. Incident sets overlap, and their deletion effects are not additive.'''),
C('''deletions=pd.read_csv(OUT/"deletions.csv")
display(deletions[deletions.kind.isin(["baseline","support","network_incident"])][["kind","removed","union_edges_removed","primary_delta","pca_delta","reversal_shift"]])
display(deletions[deletions.kind=="top_edges"][["removed","primary_delta","pca_delta","reversal_shift"]])
display(Image(filename=str(RESULTS/"figures/edge_driver_deletions.png")))'''),
M('''## Replay the same motion/duration-matched samples
All 6×200 previous draws are reproduced using exactly their saved masks, both frozen and fully refitted nuisance models. Per-edge contrasts sum to each prior global matched Δ. Support/pair **means** sum to the mean total; marginal medians/quantiles generally do not. Ranges remain conditional sampling spreads, not confidence intervals or probabilities of a surgery effect.

For strict pools, the pre-only full-refit term is fixed at **−0.03102**, because all 68 pre-op frames are selected on every draw. Changing FD-only matching to FD+run-half matching shifts the **shared-edge mean term from −0.03272 to −0.01395**, and the post-only mean term from +0.05032 to +0.05560. The total mean changes from −0.01343 to +0.01063. Most of that design difference comes from shared edges, rather than changing the fixed pre-only term.

A separate `refit_effect` is Δ(refitted)−Δ(frozen) on the **same selected frames**; it is not the original strict-minus-primary full-sample R. The pre-only effect is +0.01782 in all strict designs, while shared effects change from −0.00489 under FD matching to +0.01224 with half matching. Across network-pair means, the largest positive half-versus-FD design shift is **Cont–SomMot +0.00408**, whereas that pair opposes the original full-sample R. Driver localization therefore depends on the question and selected data, not only the atlas.'''),
C('''matched_groups=pd.read_csv(OUT/"matched_group_summary.csv")
replay=pd.read_csv(OUT/"matched_reproduction.csv")
print("Largest discrepancy from previous matched Δ:",replay.previous_delta_error.abs().max())
def matched_components(protocol="strict_motion_time",mode="refit",partition="support"):
    g=matched_groups[(matched_groups.protocol==protocol)&(matched_groups["mode"]==mode)&(matched_groups.partition==partition)].sort_values("mean_contribution")
    display(g[["group","edges","mean_contribution","median_contribution","q025","q975","positive_fraction"]])
    fig,ax=plt.subplots(figsize=(9,max(3,.22*len(g))))
    ax.barh(g.group,g.mean_contribution,color="#245a81");ax.axvline(0,color="black",lw=.7)
    ax.set_xlabel("Mean conditional post − pre contribution" if mode!="refit_effect" else "Mean same-frame refitting contribution")
    ax.set_title(f"{protocol}: {mode}; incident-network sets overlap");plt.tight_layout();plt.show();plt.close(fig)
widgets.interact(matched_components,protocol=widgets.Dropdown(options=matched_groups.protocol.unique().tolist(),value="strict_motion_time"),mode=widgets.Dropdown(options=["frozen","refit","refit_effect"],value="refit"),partition=widgets.Dropdown(options=["support","network_pair","network_incident"],value="support"))'''),
M('''### Follow one edge through the refitting sequence
The edge inventory retains measured FC even where SC is absent, but absent edges make no contribution to that session's coupling. Inspect Pearson FC alongside the edge's *global-rank* contribution. A large FC change is not necessarily a large coupling contribution, and an additive rₑ is not the same as its leave-one-out deletion effect.'''),
C('''ranked=edges.reindex(edges.reversal_shift.abs().sort_values(ascending=False).index).reset_index(drop=True)
def inspect_edge(rank=1):
    r=ranked.iloc[rank-1]
    display(r[["node_i","node_j","label_i","label_j","support_group","sc_pre","sc_post","reversal_shift","leave_one_primary_delta","leave_one_pca_delta"]])
    stages=["primary","restriction","coefficients","pca"]
    fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    for session,color in [("preop","#245a81"),("postop","#b35322")]:
        axs[0].plot(range(4),[r[f"{session}_{s}_fc"] for s in stages],"o-",color=color,label=session)
        axs[1].plot(range(4),[r[f"{session}_{s}_contribution"] for s in stages],"o-",color=color,label=session)
    for ax in axs:ax.set_xticks(range(4),["Primary","Restrict","Refit coeff.","Refit PCA"],fontsize=8);ax.legend()
    axs[0].set_ylabel("Measured edge FC (Pearson r)");axs[1].set_ylabel("Contribution to own-support Spearman ρ");plt.tight_layout();plt.show();plt.close(fig)
widgets.interact(inspect_edge,rank=widgets.IntSlider(min=1,max=len(ranked),value=1,continuous_update=False))'''),
M('''## Reproducibility and interpretation
`uv run python scripts/investigate_edge_drivers.py` regenerates the exact decomposition, complete 2003-edge CSV inventory, every single-edge deletion, support/network stress tests and matched replay. No imaging/FSL stage runs. All masks, counts, atlas IDs, measured FC, rank terms and source hashes are bundled; `matched_edge_contrasts.npz` stores draw×edge arrays in `edges.csv` order. See `config/COUPLING_EDGE_DRIVERS.md`.

This analysis localizes a **methodological sensitivity**. Pre-only edges dominate the original signed rank accounting, Default cross-network pairs provide the largest positive totals, and opposing terms are substantial. Reranked deletion and matched controls change that account. None establishes tract loss, neural recovery/decline, or a cognitive/QoL mechanism. This original edge accounting remains a seed-160 sensitivity. New common-consensus edge/network accounting is in `results/primary_sc`; all clinical source fields remain unchanged.

**Exercises:** explain why a small post-only additive R term can coexist with a large deletion effect; reconstruct R from support and pair sums; compare a top individual edge with its pair's aggregate sign; compare shared-edge pooled contributions with shared-only Spearman; identify which matched terms remain fixed because pre-op strict samples do not change. Continue in notebook [09](09_structural_edge_reliability.ipynb) for repeated tractography and count-threshold controls. What repeated imaging and physiological evidence would be needed before assigning a biological interpretation?''')])
write('09_structural_edge_reliability.ipynb','# 09 | How reliable are low-count and session-specific SC edges?\n\n**Learning goals:** separate seed-sampling stability from anatomical validity; inspect newly tracked streamlines; compare count thresholds, consensus supports and coupling; preserve the distinction between SC and FC uncertainty.',[
M('''## What is repeated, and what stays fixed?
Twenty **new tractography runs per session**, each with 30,000 seed locations, use independent RNG initializations (17000–17019 pre-op; 18000–18019 post-op). CSD directions, diffusion data, masks, geometry, stopping and endpoint rules stay fixed. This is random seed-location variation in deterministic tracking, **not resampling the original tractogram**, new diffusion scans, diffusion-noise bootstrapping, registration variation or anatomical validation.

The original tracker used float64 CSD coefficients but saved a float32 cache. The identical high-shell model/response is refitted once in float64; separate seed-160 replays must reproduce both original count matrices exactly. Original SC, FC, nuisance fits and previous sensitivity outputs are retained. All 4,950 cortical pairs are evaluated, including edges absent from both original runs. “Session-specific” means a particular reconstruction's support, not proven tract loss/gain.'''),
C('''from pat16.tract_repeats import OUT as REL, REPEATS, SEEDS
from pat16.sc_reliability import load, threshold_weights, consensus_weights, evaluate, THRESHOLDS, CONSENSUS
ri,rj,baseline_sc,repeat_sc,reliability_fc=load()
reliable_edges=pd.read_csv(REL/"edges.csv")
reliability_summary=pd.read_csv(REL/"edge_reliability_summary.csv")
fixed_reliability=pd.read_csv(REL/"fixed_coupling.csv")
repeat_summary=pd.read_csv(REL/"repeat_coupling_summary.csv")
for session in SESSIONS:
    print(session,json.loads((REL/session/"baseline_replay.json").read_text()))
display(reliability_summary[reliability_summary.baseline_count_bin=="all"])
display(Image(filename=str(RESULTS/"figures/sc_repeat_detection.png")))
display(Image(filename=str(RESULTS/"figures/sc_repeat_counts.png")))'''),
M('''## Inspect occurrence and count variability for any edge
Occurrence at k is the fraction of 20 runs with **count ≥ k**, at the same 30,000-seed budget (resolution 0.05). Count SD/CV include zero-count runs; all-zero means have undefined CV and are omitted from median CV. Original singletons have median occurrence 0.55/0.50 (pre/post), versus 1.00/1.00 for original counts ≥10. The original pre-only/post-only labels are fixed; inspect whether those edges reappear in the opposite session. No hits in 20 runs is limited sampling evidence, not confirmed anatomical absence.

Start with edge 21–42, the largest positive original reversal term: it appears in only 3/20 pre-op and 2/20 post-op repeats (mean count 0.15 in each session). Compare it with the largest negative term 16–35, and with high-count edges. Atlas centroids/labels do not validate an individual connection. Thresholds apply to individual runs, not to the pooled count over 600,000 seeds.'''),
C('''def inspect_reliability(node_a=21,node_b=42,minimum_count=1):
    if node_a==node_b:
        print("Choose two different parcels.");return
    a,b=sorted([node_a,node_b]);idx=int(np.flatnonzero((ri==a-1)&(rj==b-1))[0]);row=reliable_edges.iloc[idx]
    display(row[["node_i","node_j","label_i","label_j","baseline_group","repeat_consensus80_class","original_reversal_term"]].to_frame("value"))
    fig,axs=plt.subplots(1,2,figsize=(11,3.8))
    for t,session in enumerate(SESSIONS):
        vals=repeat_sc[t,:,idx];freq=float((vals>=minimum_count).mean())
        axs[t].bar(np.arange(1,21),vals,color="#245a81" if t==0 else "#b35322")
        axs[t].axhline(baseline_sc[t,idx],color="black",ls="--",label="original seed 160")
        axs[t].axhline(minimum_count,color="#888",ls=":",label="inclusive count cutoff")
        axs[t].set_title(f"{session}: detected {int((vals>=minimum_count).sum())}/20 = {freq:.2f}")
        axs[t].set_xlabel("New tracking run");axs[t].set_ylabel("Assigned streamline count");axs[t].legend(fontsize=8)
    plt.tight_layout();plt.show();plt.close(fig)
widgets.interact(inspect_reliability,node_a=widgets.IntSlider(min=1,max=100,value=21,continuous_update=False),node_b=widgets.IntSlider(min=1,max=100,value=42,continuous_update=False),minimum_count=widgets.Dropdown(options=THRESHOLDS,value=1))'''),
M('''## Count thresholds change the ranked edge universe
We test inclusive minimum counts **1, 2, 3, 5, 10, 20** without choosing a cutoff from its coupling sign. “Own” thresholds each session separately; “common” retains the intersection at that cutoff with each session's own count weights. Measured FC is unchanged; SC-absent edges are excluded, never replaced by FC=0. Every SC/FC rank and denominator is recomputed.

For each threshold, all 20×20 pre/post repeat combinations are considered rather than arbitrarily pairing distinct RNG IDs. The 400 combinations **reuse runs and are dependent**. Empirical ranges and sign fractions describe seed sensitivity conditional on fixed scans/FC, not confidence intervals, probabilities of a surgical effect or 400 independent subjects. At k=1 with own repeated SC, median Δ is −0.00043/+0.02852 for primary/strict, and the original negative-primary/positive-strict pattern occurs in 174/400 combinations. Both empirical ranges cross zero. Thresholding can change both contrast signs while preserving a nonzero strict-minus-primary shift R; these are different questions. Repeated common-support strict median is +0.00579, even though the original common-support strict contrast is negative; a single shared-only result is also seed-conditional.'''),
C('''def threshold_browser(minimum_count=1,support="own"):
    a=fixed_reliability[(fixed_reliability.method=="baseline")&(fixed_reliability.threshold==minimum_count)&(fixed_reliability.support==support)]
    b=repeat_summary[(repeat_summary.threshold==minimum_count)&(repeat_summary.support==support)]
    display(a[["threshold","support","pre_edges","post_edges","shared_edges","primary_pre","primary_post","primary_delta","strict_pre","strict_post","strict_delta","reversal_shift"]])
    display(b)
    terms=pd.read_csv(REL/"coupling_contributions.csv")
    terms=terms[(terms.method=="baseline")&(terms.threshold==minimum_count)&(terms.support==support)&(terms.partition=="network_pair")]
    display(terms.sort_values("reversal_term",ascending=False).head(5))
widgets.interact(threshold_browser,minimum_count=widgets.Dropdown(options=THRESHOLDS,value=1),support=widgets.Dropdown(options=["own","common"],value="own"))
display(Image(filename=str(RESULTS/"figures/sc_repeat_coupling.png")))'''),
M('''## Consensus support is another sensitivity analysis
At occurrence fractions 0.5, 0.8, 0.95 or 1.0, retain edges whose per-run count reaches k often enough. SC weights are **mean counts per 30,000 seeds**, not occurrence probabilities or pooled-count thresholds. Compare own versus common consensus supports. Consensus can admit an edge missing from the original run and exclude an edge seen only by chance; it also changes the population and weighting of coupling.

At k=1, 80% own consensus retains 1160/1163 edges and yields primary/strict Δ=−0.00257/+0.02363. Common consensus retains 823 edges and gives −0.04173/−0.03424. Neither validates a directional biological change.

The 80% class “pre-only” only means the post-op frequency is below 0.8; it can still be detected in many post-op runs. A separate inventory flag marks ≥0.8 in one session with 0/20 in the other. Neither proves loss/gain. Mean SC uses all 20 runs and is not independent anatomical validation. The consensus fraction and streamline cutoff are exploratory controls, not calibrated false-positive thresholds.'''),
C('''reliability_terms=pd.read_csv(REL/"coupling_contributions.csv")
def consensus_browser(minimum_count=1,occurrence=.8,support="common"):
    w=consensus_weights(repeat_sc,minimum_count,occurrence,support)
    display(pd.DataFrame([evaluate(w,reliability_fc)]))
    selected=reliability_terms[(reliability_terms.method=="consensus")&(reliability_terms.threshold==minimum_count)&np.isclose(reliability_terms.consensus_fraction,occurrence)&(reliability_terms.support==support)]
    display(selected[selected.partition=="repeat_consensus80"])
    display(selected[selected.partition=="network_pair"].sort_values("reversal_term",ascending=False).head(5))
    fig,axs=plt.subplots(1,2,figsize=(9,3.5))
    for t,session in enumerate(SESSIONS):
        mat=np.zeros((100,100));mat[ri,rj]=w[t]>0;mat+=mat.T
        axs[t].imshow(mat,cmap="Blues",vmin=0,vmax=1);axs[t].set_title(session+": retained consensus support");axs[t].set_xlabel("Atlas node");axs[t].set_ylabel("Atlas node")
    plt.tight_layout();plt.show();plt.close(fig)
widgets.interact(consensus_browser,minimum_count=widgets.Dropdown(options=THRESHOLDS,value=1),occurrence=widgets.Dropdown(options=CONSENSUS,value=.8),support=widgets.Dropdown(options=["own","common"],value="common"))
display(Image(filename=str(RESULTS/"figures/sc_repeat_consensus.png")))'''),
M('''## Revisit the original reversal accounting
The original R contributions from notebook 08 can be cross-tabulated by 80% repeat-consensus class without changing those original ranks. These signed terms still sum to the original R. This distinguishes an original “pre-only” edge from an edge that is repeatedly supported in both sessions.

Of 444 original pre-only edges, 322 reappear at least once post-op; 342 of 488 original post-only edges reappear pre-op. Only 31/24 reach ≥80% occurrence in those previously absent sessions. The original pre-only reversal term is concentrated in **207 edges below 80% occurrence in both sessions**, contributing **+0.02412** of the full +0.03186 shift. These are signed cancelling terms, not percentages of neural variance.

The re-ranked terms for new threshold/consensus matrices answer a different question. The browser above exports their network-pair and support contributions as well. Do not subtract original low-count edge terms and call that thresholded coupling: all surviving ranks change.'''),
C('''cross=pd.read_csv(REL/"original_reversal_by_reliability.csv")
display(cross.pivot(index="baseline_group",columns="repeat_consensus80_class",values="edges").fillna(0).astype(int))
display(cross.pivot(index="baseline_group",columns="repeat_consensus80_class",values="original_reversal_term").fillna(0))
print("Original R reconstructed:",cross.original_reversal_term.sum())
for group,other in [("pre_only","postop"),("post_only","preop")]:
    v=reliable_edges[reliable_edges.baseline_group==group]
    print(group,"opposite-session rediscovery:",int((v[other+"_frequency_ge1"]>0).sum()),"of",len(v),"; opposite-session >=80%:",int((v[other+"_frequency_ge1"]>=.8).sum()))'''),
M('''## Separate FC sampling from the SC reference
All six earlier matching protocols ×200 saved masks are replayed with full nuisance refitting. Five SC references are held fixed: original own SC at counts ≥1, ≥3, ≥5 and 80% consensus at k=1 under own/common support. The original k=1 results must reproduce notebook 07 exactly. Historical frozen-nuisance analyses remain there.

These spreads vary the selected BOLD frames while holding SC fixed. The earlier repeat plots vary SC seeding while holding FC fixed. **Notebook 10 now crosses whole-run SC bootstrap draws with FC draws under the recommended policy**, including fixed and reselected support. These historical one-source spreads are retained and must not be added to manufacture a joint interval.'''),
C('''matched_rel=pd.read_csv(REL/"matched_coupling_summary.csv")
def matched_reference(protocol="strict_motion"):
    v=matched_rel[matched_rel.protocol==protocol].reset_index(drop=True)
    display(v)
    fig,ax=plt.subplots(figsize=(9,3.5));y=np.arange(len(v))
    ax.errorbar(v["median"],y,xerr=np.vstack([v["median"]-v.q025,v.q975-v["median"]]),fmt="o",capsize=3,color="#245a81")
    ax.axvline(0,color="#777",ls="--");ax.set_yticks(y,v.sc_reference);ax.set_xlabel("Post − pre coupling; median and conditional 2.5/97.5 percentiles");ax.set_title(protocol+": fixed SC, fully refitted matched FC")
    plt.tight_layout();plt.show();plt.close(fig)
widgets.interact(matched_reference,protocol=widgets.Dropdown(options=matched_rel.protocol.unique().tolist(),value="strict_motion"))'''),
M('''## Conditional repeatability and what remains unknown
Within-session Jaccard measures overlap of thresholded supports; count-rank agreement is computed over all 4,950 edges, including tied zeros. Split-half controls select 80% consensus in the first ten runs and inspect occurrence in the last ten, then reverse the roles. This checks seeding consistency within this model; systematic false positives can repeat, and model, registration, endpoint and acquisition errors are held fixed.

The primary recommendation is now 16/20 consensus with all-run mean counts per 10,000 seeds. Structural graphs use own consensus (1160/1163 edges); paired coupling uses its fixed 823-edge intersection. Original seed-160 and all other count/consensus settings remain sensitivities. The 80% cutoff is pragmatic, not calibrated; it is selected for repeat stability and comparable edge populations, never a desired coupling sign. These results cannot establish tract loss/gain or localize CANTAB/QoL changes. Independent repeated acquisitions, lesion-aware mapping, alternative tracking/model choices and anatomical review would address additional uncertainties.'''),
C('''overlap_rel=pd.read_csv(REL/"within_session_overlap.csv")
display(overlap_rel.groupby(["session","threshold"]).agg(median_Jaccard=("jaccard","median"),q025_Jaccard=("jaccard",lambda x:x.quantile(.025)),q975_Jaccard=("jaccard",lambda x:x.quantile(.975)),median_count_rank=("count_spearman_all_4950_edges","median")))
display(pd.read_csv(REL/"split_half.csv"))
print(json.loads((REL/"summary.json").read_text()))'''),
M('''## Reproduce and investigate
Run `uv run python scripts/investigate_sc_reliability.py` to reuse or generate all tracking runs, analyses and figures. No FSL stage is repeated. The run directory contains float64 CSD coefficients, per-run seed locations, endpoint pairs/distances/lengths, count matrices, QC and input/code hashes. Full specifications are in `config/SC_RELIABILITY.md`.

**Exercises:** compare detection at baseline counts 1 versus 10; find an original pre-only edge frequently rediscovered post-op; distinguish 80% own from common support; compare thresholded seed-160 coupling with the 400 repeat combinations; explain why a stable false-positive pathway can survive consensus; compare fixed-SC matched FC with fixed-FC repeated SC. What independent evidence would be needed before calling an edge surgically disconnected?''')])
write('10_joint_sampling_uncertainty.ipynb','# 10 | Joint tractography and FC sampling uncertainty\n\n**Learning goals:** resample the 20-run SC estimator; preserve edge dependence; distinguish fixed support from selecting it again; cross SC with temporal and fully refitted matched FC; interpret conditional ranges without making a surgical claim.',[
M('''## What is being resampled?
The recommended primary statistic is Spearman SC–FC coupling on the **823 common consensus edges**, with separate pre/post all-run mean SC weights per 10,000 seeds and anatomical-five-PC FC. Observed post−pre Δ is −0.04173 at primary FD and −0.03424 at strict FD. These point estimates stay unchanged.

We independently bootstrap **20 whole tractography runs with replacement** within each session, 500 times. This resamples the *20-run mean*, including zero counts, rather than substituting one tractogram. All edges in a run move together. For each of 200 paired FC draws, all cortical ROI signals use the same frame indices. We cross all SC and FC draws: 100,000 **dependent** cells per protocol, not 100,000 scans or independent subjects.

**Fixed support** locks the observed 823 connections and varies their mean weights. **Reselected support** reapplies detection ≥16/20 independently in both sessions and uses their intersection, reranking SC and FC each time. This includes support-selection sensitivity but changes the ranked edge population; it is not the identical fixed-population estimand. The 80% rule and all other primary choices remain fixed, not optimized by sign.'''),
C('''from pat16.joint_sampling import OUT as JOINT, I as ji, J as jj
joint_design=json.loads((JOINT/"design.json").read_text())
joint_summary=pd.read_csv(JOINT/"summary.csv")
support_qc=pd.read_csv(JOINT/"support_qc.csv")
fc_qc=pd.read_csv(JOINT/"fc_qc.csv")
display(pd.Series({k:joint_design[k] for k in ["version","SC_draws","FC_draws","cells_per_crossed_grid","independent_grid_cells"]}))
session_coupling=pd.read_csv(JOINT/"session_coupling.csv")
display(session_coupling[(session_coupling.protocol=="primary_block10")&(session_coupling.source=="joint")][["session","support","observed_rho","median","q025","q975"]])
display(Image(filename=str(RESULTS/"figures/joint_sampling_ranges.png")))'''),
M('''## Historical frozen full-run branch; see notebook 11 for refits
Temporal sampling and nuisance training are different
Full-run temporal resampling draws circular blocks of **10 original frames (24 s)**, omits censored members *within each block*, and continues until the original retained count is reached (146/150 primary, 68/96 strict). We never concatenate good frames before drawing blocks. All 100 ROI residuals move together. The session's censor-specific nuisance model is **frozen**: these ranges condition on its training. Blocks of 5/20 frames (12/48 s) test this arbitrary temporal scale. Repeated indices reduce distinct observations; the original regression df must not be assigned to them. Circular stationarity and within-run representativeness are assumptions.

The six **matched protocols** use the saved 68-distinct-frame masks: duration, duration+FD and duration+FD+run-half in primary/strict eligible pools. Tissue temporal projection, scaling, five PCs and regression are **fully refitted** on every subset, on the original 180-frame basis. These are subsampling designs without replacement, not block-bootstrap CIs. FD pairs use the existing 0.015-mm caliper. Strict pre-op has exactly 68 eligible frames: **its matched FC is constant**, and matched FC variability is postoperative only.

Matched sampling and full-run block sampling answer different questions. Neither provides scan–rescan error or a calibrated interval for a surgical effect. See `config/JOINT_SAMPLING.md`; circular blocks are motivated by temporal dependence (Bellec et al., 2008, report bibliography), whose test calibration is not asserted for this censored rank statistic.'''),
C('''def joint_ranges(protocol="primary_block10",support="fixed"):
    v=joint_summary[(joint_summary.protocol==protocol)&(joint_summary.support==support)]
    display(v[["source","observed_delta","median","q025","q975","sd","positive_fraction"]])
    fig,ax=plt.subplots(figsize=(8,3));y=np.arange(len(v))
    ax.errorbar(v["median"],y,xerr=np.vstack([v["median"]-v.q025,v.q975-v["median"]]),fmt="o",capsize=4)
    ax.axvline(0,color="gray",ls="--");ax.set_yticks(y,v.source)
    ax.set_xlabel("Post − pre Δ: conditional median and 2.5/97.5 percentiles")
    ax.set_title(protocol+"; "+support+" support");plt.tight_layout();plt.show();plt.close(fig)
widgets.interact(joint_ranges,protocol=widgets.Dropdown(options=joint_design["protocols"],value="primary_block10"),support=widgets.Dropdown(options=["fixed","reselected"]))'''),
M('''## Inspect an actual crossed cell
SC and FC draws are independent computational axes; pre/post run numbers are never assumed to correspond. Each saved grid stores both session correlations and Δ. FC-only holds the point SC/support fixed. SC-only holds the *observed full-run FC* fixed, even for a matched protocol: it is an observed-data anchor, not the SC main effect after averaging over matched FC. The joint grid evaluates both perturbations together and reranks on the actual support. Percentiles of marginal sources cannot be added.

Positive fractions are empirical sensitivity summaries. The reused rows/columns make binomial error bars based on 100,000 cells invalid, and a fraction is not a probability of neural improvement.'''),
C('''fc_vectors=np.load(JOINT/"fc_draws.npy",mmap_mode="r")
sc_vectors=np.load(JOINT/"sc_weights.npy",mmap_mode="r")
selected_support=np.load(JOINT/"sc_selected_support.npy",mmap_mode="r")
fixed_support=np.load(RESULTS/"primary_sc/preop/common_support.npy")[ji,jj]
def crossed_cell(protocol="primary_motion",support="fixed",sc_draw=0,fc_draw=0):
    with np.load(JOINT/f"{protocol}_{support}.npz") as z:
        delta=z["delta"];r=z["rho"][:,sc_draw,fc_draw]
        display(pd.Series({"preop_rho":r[0],"postop_rho":r[1],"delta":r[1]-r[0],"SC_draw":sc_draw,"FC_draw":fc_draw,"edges":int(fixed_support.sum() if support=="fixed" else selected_support[sc_draw].sum())}))
        fig,axs=plt.subplots(1,2,figsize=(10,3.5))
        axs[0].imshow(delta,aspect="auto",cmap="coolwarm",vmin=-.15,vmax=.15)
        axs[0].plot(fc_draw,sc_draw,"o",ms=5,mfc="none",mec="black")
        axs[0].set(xlabel="FC draw",ylabel="SC draw",title="Crossed Δ (fixed display scale)")
        axs[1].hist(delta.ravel(),bins=45,color="#346d9d");axs[1].axvline(0,color="gray",ls="--")
        axs[1].axvline(r[1]-r[0],color="#c88b39");axs[1].set(xlabel="Post − pre Δ",ylabel="Dependent cells")
        plt.tight_layout();plt.show();plt.close(fig)
    display(fc_qc[(fc_qc.protocol==protocol)&(fc_qc.draw==fc_draw)])
widgets.interact(crossed_cell,protocol=widgets.Dropdown(options=joint_design["protocols"],value="primary_motion"),support=widgets.Dropdown(options=["fixed","reselected"]),sc_draw=widgets.IntSlider(min=0,max=499,value=0),fc_draw=widgets.IntSlider(min=0,max=199,value=0))'''),
M('''## Where does the conditional grid variance come from?
For Δ(b,f), subtract its grand mean. The SC main effect is the centered mean of each row, the FC main effect is the centered mean of each column, and interaction is what remains. With equal product weights these are orthogonal; their population variances sum **exactly** to the grid variance. This is a description of the chosen independent computational perturbations, not identification of independent biological errors.

Support re-selection changes both the connection population and the rank context. Its variance can be larger or smaller than fixed-support variance; ranges need not widen monotonically. Mean counts and consensus are selected from only 20 empirical seed runs, and repeatedly false pathways remain possible.'''),
C('''joint_components=joint_summary[joint_summary.source=="joint"]
display(joint_components[["protocol","support","SC_share","FC_share","interaction_share","total_variance"]])
display(Image(filename=str(RESULTS/"figures/joint_sampling_components.png")))
display(support_qc[["common_edges","added_edges","removed_edges","jaccard"]].describe(percentiles=[.025,.5,.975]))
selection=pd.read_csv(JOINT/"edge_selection.csv")
display(selection[selection.point_common].sort_values("bootstrap_common_frequency").head(15))'''),
M('''## Temporal scale, finite Monte Carlo sampling and matching checks
Block length is not fitted to this case. Compare 5/10/20 frames under both censor pools; inspect matched protocols separately. Half-ensemble ranges reveal finite simulation sensitivity, especially in the tails. In the reselected strict FD-only design, the upper endpoint crosses zero in one FC half and remains negative in the other, so its small full-ensemble negative endpoint is not a stable directional result. The exact values are in the table below. They are not independent validation or a convergence proof. The six saved matching protocols and their motion/temporal QC remain in notebook 07. Every matched frame count is 68, regression rank 39, residual df 29; block draws have no newly fitted nuisance rank or df.'''),
C('''blocks=joint_summary[(joint_summary.source=="joint")&joint_summary.protocol.str.contains("block")]
display(blocks[["protocol","support","median","q025","q975","positive_fraction"]])
display(pd.read_csv(JOINT/"monte_carlo_halves.csv")[["protocol","support","half","median","q025","q975"]])
display(fc_qc.groupby(["protocol","session","nuisance"]).agg(retained=("retained","min"),unique_min=("unique_frames","min"),unique_median=("unique_frames","median"),unique_max=("unique_frames","max"),rank=("design_rank","first"),df=("residual_df","first")))'''),
M('''## Reproduce and interpret
Run `uv run python scripts/investigate_joint_sampling.py`, then `uv run python scripts/run_all.py --notebooks --report`. The new analysis is separate under `results/joint_sampling/`: all SC run indices, weights, supports, FC matrices, block indices, matched masks, QC and 24 crossed grids are saved. `design.json` fingerprints source/config/input/output files. The preceding delivery is preserved under `results/before_joint_sampling/`.

The point estimates and recommended policy are unchanged. Conditional seed and FC sampling must be assessed together, while the full-run branch still conditions on nuisance training and the matched branch still conditions on measured motion and short eligible pools. Acquisition, image noise, registration, tissue masks, model choice, vascular changes and repeated clinical assessments are not resampled. None of these ranges establishes a surgical, cognitive or quality-of-life effect.''')
])



write('11_full_run_nuisance_refit.ipynb','# 11 | Full-run sampling with nuisance training refitted\n\n**Question:** How much changes when each FC draw retrains its WM/CSF nuisance model? Compare exactly paired frozen, coefficient-only and complete refits, on fixed and reselected SC support.',[
M('''## One draw, three estimators
Reuse all saved original-time blocks and all 500 whole-run SC bootstraps from notebook 10. **Frozen** samples the old residuals. **Coefficients** fixes the censor-pool's five anatomical PC scores but refits regression coefficients and scaling. **Full refit** also retrains tissue temporal projection, voxel centering/scaling and five-PC extraction. Spatial masks, image correction, ROI signals, motion and censor eligibility stay fixed.

Multiplicity weights give the same estimator as explicitly repeating sampled rows. Derivatives and DCT time coordinates remain on the original 180-frame axis: artificial block joins do not become motion events. All ROI signals move together. The crossed cells share rows and columns and are dependent; these are conditional computational ranges, not clinical confidence intervals.'''),
C('''from pat16.joint_refit import OUT as REFIT, PROTOCOLS as RP, MODES as RM
refit_design=json.loads((REFIT/"design.json").read_text())
refit_summary=pd.read_csv(REFIT/"summary.csv")
fit_qc=pd.read_csv(REFIT/"fit_qc.csv")
feasibility=pd.read_csv(REFIT/"feasibility.csv")
display(feasibility)
display(Image(filename=str(RESULTS/"figures/joint_refit_ranges.png")))'''),
M('''## Duplicates are frequency, not new measured information
Every requested draw is attempted. We explicitly impose the conservative feasibility safeguards **≥60 distinct original frames** and **≥15 distinct rows minus design rank**, with agreement between design and solver rank. This extends the existing fit safeguards to distinct bootstrap rows; it is not a universal bootstrap theorem. Neither count-minus-rank estimates independent BOLD information. Save failed-draw QC and NaN FC, never replace failures.

Compare all three models only on the same eligible paired draws. A range requires at least 100 of 200 planned pairs, chosen before computing the contrasts. Always inspect the counts; rejection can change the conditional population. Strict pre-op starts with only 68 retained frames, so block repetitions can leave too few distinct rows for the 39-column model. Withholding its range is a result, not evidence of a stable sign. Notebook 10 retains the original all-draw frozen results, and its six distinct-frame matched designs remain separate fully refitted sensitivities.'''),
C('''display(Image(filename=str(RESULTS/"figures/joint_refit_feasibility.png")))
display(fit_qc.groupby(["protocol","session","mode"]).agg(attempted=("draw","count"),eligible=("eligible","sum"),unique_min=("unique_frames","min"),unique_median=("unique_frames","median"),unique_max=("unique_frames","max"),minimum_distinct_residual=("distinct_rows_minus_rank","min"),median_PC_similarity=("minimum_PC_subspace_cos2","median")))'''),
C('''def compare_refits(protocol="primary_block10",support="fixed"):
    display(feasibility[feasibility.protocol==protocol])
    v=refit_summary[(refit_summary.protocol==protocol)&(refit_summary.support==support)]
    if v.empty:
        display(Markdown("**Range withheld:** fewer than 100 paired eligible draws. Inspect fit QC and the historical frozen branch in notebook 10."));return
    display(v[["mode","FC_draws","median","q025","q975","positive_fraction","SC_share","FC_share","interaction_share"]])
    shifts=pd.read_csv(REFIT/"paired_shifts.csv")
    display(shifts[(shifts.protocol==protocol)&(shifts.support==support)])
    fig,ax=plt.subplots(figsize=(8,3));y=np.arange(len(v))
    ax.errorbar(v["median"],y,xerr=np.vstack([v["median"]-v.q025,v.q975-v["median"]]),fmt="o",capsize=4)
    ax.axvline(0,color="gray",ls="--");ax.set_yticks(y,v["mode"]);ax.set_xlabel("Post − pre coupling Δ; conditional 2.5/97.5 percentiles")
    plt.tight_layout();plt.show();plt.close(fig)
widgets.interact(compare_refits,protocol=widgets.Dropdown(options=list(RP),value="primary_block10"),support=widgets.Dropdown(options=["fixed","reselected"]))'''),
M('''## Inspect the actual draw and the two refitting steps
For a paired cell, Δ(full)−Δ(frozen) equals the sum of the coefficient-only shift and the further PC-training shift. This is ordered computational accounting, not independent causal sources. Subspace cosine² compares retained PC spans after multiplicity weighting; arbitrary PC signs are irrelevant. The full-refit estimator can change both the center and width of a range. Refitting need not simply widen a frozen range.'''),
C('''refit_fc=np.load(REFIT/"fc_draws.npy",mmap_mode="r")
def inspect_refit(protocol="primary_block10",draw=0):
    display(fit_qc[(fit_qc.protocol==protocol)&(fit_qc.draw==draw)])
    p=list(RP).index(protocol);v=refit_fc[p,:,:,draw]
    if not np.isfinite(v).all():
        print("This pair fails feasibility; unavailable FC is saved as NaN, not zero.");return
    from scipy.stats import spearmanr
    weights=np.load(RESULTS/"joint_sampling/sc_weights.npy",mmap_mode="r")[:,0]
    ii,jj=np.triu_indices(100,1);mask=np.load(RESULTS/"primary_sc/preop/common_support.npy")[ii,jj]
    r=np.array([[spearmanr(weights[t,mask],v[m,t,mask]).statistic for t in range(2)] for m in range(3)])
    display(pd.DataFrame({"mode":list(RM),"preop":r[:,0],"postop":r[:,1],"delta":r[:,1]-r[:,0]}))
widgets.interact(inspect_refit,protocol=widgets.Dropdown(options=list(RP),value="primary_block10"),draw=widgets.IntSlider(min=0,max=199,value=0))
display(pd.read_csv(REFIT/"monte_carlo_halves.csv"))'''),
M('''## Reproduce and extend
Run `uv run python scripts/investigate_joint_refit.py`. `results/joint_refit/design.json` fingerprints code, configuration, saved draws and outputs. Exact repeated-row PCA/WLS equivalence, unchanged full-run point FC, frozen replay, rank/range identities and preservation of the preceding delivery are tested.

**Exercises:** compare block sizes 5/10/20; separate coefficient refitting from PC retraining; explain why an interval can shift without widening; inspect a failed strict pre-op draw; distinguish numerical rank from independent BOLD information; compare these full-run results with the 68-distinct-frame matched controls. Notebook 12 connects this evidence to the assessments without fitting two-point clinical associations. Notebook [13](13_pc_sampling_mechanisms.ipynb) separates tissue preprocessing from PCA-loading refits and shows that repeated-frame weights drive much of the recentering; it compares block subsampling without replacement. Empirical distribution shifts do not identify population bias.''')])

write('12_connectivity_cognition_context.ipynb','# 12 | Integrate connectivity uncertainty, cognition and context\n\n**Goal:** examine the complete clinical profile alongside measured network changes and conditional uncertainty, without reducing the patient to one network metric or an invented cognitive score.',[
M('''## Four evidence layers
1. Raw clinical values and changes: 28 CANTAB measures and 19 psychosocial/context fields, with original units, missingness and coding caveats.
2. Observed SC, FC and SC–FC point summaries, under the recommended fixed policy.
3. Conditional repeat/sampling ranges and frozen-versus-refitted nuisance sensitivity.
4. Candidate task/network hypotheses, retained from notebook 04, **not individual localization or fitted prediction**.

Context fields have no assigned brain network. Faster RTI and longer SSP coexist with slower MOT, mixed SOC/RVP and mixed emotional/social outcomes. Lower FC is not uniformly worse cognition. No dedicated quality-of-life instrument, clinical repeat-error estimate, norm-adjusted reliable-change threshold or sufficient observations for a clinical association model is supplied.'''),
C('''EVIDENCE=RESULTS/"evidence_integration"
ledger=pd.read_csv(RESULTS/"primary_fc/clinical_evidence_ledger.csv")
network_ranges=pd.read_csv(RESULTS/"primary_fc/network_uncertainty.csv").query("role=='primary'")
print("Main evidence: full-run points plus fully refitted 97-frame stability ranges. Earlier bootstrap evidence is retained in evidence_integration/.")
display(pd.read_csv(EVIDENCE/"task_network_hypotheses.csv"))
display(ledger.groupby(["group","availability"]).size().rename("fields"))
display(Image(filename=str(RESULTS/"figures/primary_fc_network_changes.png")))'''),
M('''## A field-by-field evidence browser
Select any source field. Candidate networks are broad didactic links shared across tasks. Network FC is the mean Fisher-z of **all within-network pairs**, including those without SC support; network SC is mean own-consensus weight over the same fixed anatomical pair population, including absent zero weights. SC reselects own 16/20 support in each whole-run bootstrap. Primary FC uses 200 nonreplacement 97-frame block-10 draws with complete nuisance refitting. Frozen/coefficient fits and earlier bootstrap ranges remain sensitivity evidence in notebooks 11/13/14. These network averages differ from Spearman coupling on the 823 shared SC connections; their directions need not agree.

SC and FC axes have different units and perturbation sources. Clinical changes are displayed separately in original units; no uncertainty bars are invented for them. Context scores describe reported experience/participation, and can motivate future measurement, but their causal influence here is not estimated. Post-only “Same/Less” responses remain relative categories and do not fill numeric postoperative baselines.'''),
C('''def assessment_browser(measure="rti_simple_rt_mean"):
    r=ledger[ledger.measure==measure].iloc[0]
    display(r[["measure","description","group","preop","postop","delta_post_minus_pre","units","direction_text","availability","definition","caveat","integration_role","candidate_networks","link_status"]].to_frame("value"))
    if pd.isna(r.candidate_networks):
        display(Markdown("**Unlocalized context.** Examine this field with the mixed cognitive profile and missingness; no network is assigned and no causal adjustment is possible."));return
    names=r.candidate_networks.split("; ")
    display(network_ranges[network_ranges.network.isin(names)][["network","modality","mode","units","preop","postop","observed_delta","median","q025","q975"]])
widgets.interact(assessment_browser,measure=widgets.Dropdown(options=ledger.measure.tolist(),value="rti_simple_rt_mean"))'''),
C('''def network_evidence(network="SomMot"):
    v=network_ranges[network_ranges.network==network]
    display(v)
    linked=ledger[ledger.candidate_networks.fillna("").str.split("; ").apply(lambda names:network in names)]
    display(linked[["measure","preop","postop","delta_post_minus_pre","units","caveat"]])
    fig,axs=plt.subplots(1,2,figsize=(10,3))
    for ax,mod in zip(axs,["SC","FC"]):
        w=v[v.modality==mod];y=np.arange(len(w))
        ax.errorbar(w["median"],y,xerr=np.vstack([w["median"]-w.q025,w.q975-w["median"]]),fmt="o",capsize=4)
        ax.plot(w.observed_delta,y,"x",color="black",label="Observed Δ");ax.axvline(0,color="gray",ls="--")
        ax.set_yticks(y,w["mode"]);ax.set_title(network+" "+mod);ax.set_xlabel(w.units.iloc[0]);ax.legend(fontsize=8)
    plt.tight_layout();plt.show();plt.close(fig)
widgets.interact(network_evidence,network=widgets.Dropdown(options=nodes.network.unique().tolist(),value="SomMot"))'''),
M('''## New sampling-design comparison
Notebook [13](13_pc_sampling_mechanisms.ipynb) finds that repeated-frame weights substantially change the learned PCA subspace. At block 10, removing multiplicities at identical coverage moves visual-network median Δ from −0.1993 to −0.0458, closer to the observed −0.0045; independent matched-count blocks give −0.0355. The sensorimotor matched-count range crosses zero. Dorsal-attention/default subset ranges can stay negative without establishing clinical localization. Compare all designs in original FC units below. The 80% ranges depend on retained fraction and are not full-run confidence intervals.'''),
C('''sampling_networks=pd.read_csv(RESULTS/"pc_sampling/network_ranges.csv")
from pat16.pc_sampling_figures import LABELS as SAMPLING_LABELS
def clinical_sampling_comparison(measure="rti_simple_rt_mean",mode="full_refit",block=10):
    r=ledger[ledger.measure==measure].iloc[0]
    display(r[["description","preop","postop","delta_post_minus_pre","units","availability","caveat"]].to_frame("Clinical source evidence"))
    if pd.isna(r.candidate_networks):
        display(Markdown("Context is unlocalized; no network association or QoL score is calculated."));return
    names=r.candidate_networks.split("; ")
    v=sampling_networks[sampling_networks.network.isin(names)&(sampling_networks["mode"]==mode)&(sampling_networks.protocol.str.endswith("block"+str(block)))]
    display(v[["protocol","network","observed_delta","median","q025","q975"]])
widgets.interact(clinical_sampling_comparison,measure=widgets.Dropdown(options=ledger.measure.tolist(),value="rti_simple_rt_mean"),mode=widgets.Dropdown(options=["frozen","coefficients","full_refit"],value="full_refit"),block=widgets.Dropdown(options=[5,10,20],value=10))'''),
M('''## What do these calculations change about interpretation?
The recommended primary coupling median is −0.04694 with range [−0.10261,+0.00832]; its observed full-run change remains −0.04173. Sensorimotor primary range [−0.2064,−0.0170] is negative, whereas a prior matched-count design crossed zero. Visual [−0.1594,+0.1081], control [−0.1312,+0.1283] and default [−0.1649,+0.0058] cross zero. This is sampling-design sensitivity, not a diagnosis or a clinical error estimate. The measured CANTAB/context profile is unchanged.

Full nuisance retraining adds uncertainty about the estimator, while measured CANTAB changes remain unchanged. Assess the entire profile against the network ranges: a negative sensorimotor FC contrast may coexist with both slower MOT and faster RTI because they have different motor/attention/task requirements. A small control-network mean contrast does not demonstrate preserved executive circuitry, and a span gain cannot establish compensation. Anxiety, worry, loneliness and employment move in different directions; they do not define a single QoL trajectory.

The bootstrap global primary coupling ranges include zero under frozen and full refits. Matched-count nonreplacement blocks also cross zero; 80% ranges are narrower and negative but change the conditional sampling target. Their sign does not resolve biological direction. Strict full-run refits often fail the explicit distinct-frame guard; the earlier matched duration/FD/run-half results remain conditional sensitivity evidence, with strict pre-op FC fixed by its 68 eligible frames. Do not choose denoising, support, block length or a cognitive outcome by the desired sign.

**Next measurements to test hypotheses:** dated assessments and surgery/treatment timeline; retest norms and practice/reliability information; fatigue/sleep/medication and physiological context; longer repeat resting scans; expert lesion/mask/tract review; a dedicated validated QoL/participation instrument. These are evidence needs, not inferred patient facts. No two-observation correlation, cognitive composite, reliable-change test or treatment effect is calculated.

Run `uv run python scripts/integrate_clinical_evidence.py`. The 47-row evidence ledger, per-network draws/ranges and input/output hashes are in `results/evidence_integration/`. Report Appendices A1/A2 preserve the complete field dictionaries and all source values.''')])
write('13_pc_sampling_mechanisms.ipynb','# 13 | Why does refitting nuisance PCs shift the distributions?\n\n**Learning goals:** separate frequency weighting from distinct temporal coverage, distinguish voxel preprocessing from PCA-loading refits, and compare block subsampling without interpreting a shorter-sample range as a full-run confidence interval.',[
M('''## The result and its scope
The main displacement is associated with relearning the **five-dimensional PCA loading space under repeated-frame weights**. Temporal projection alone is absorbed by existing GLM trend/DCT columns; voxel rescaling alone has little effect. At block 10, loading-only minus fixed-PC coupling contrast shifts by median **+0.02059**, versus **+0.00005** for preprocessing only. These are paired computational interventions, not independent biological mechanisms.

The shift largely disappears when the bootstrap's exact distinct frames each receive unit weight. Independent nonoverlapping block subsets at matched distinct counts give similar global coupling centers. More retained frames reduce sensitivity, but their smaller ranges are conditional on the retained fraction. Neither a displaced bootstrap median nor a negative subsample range replaces the observed primary Δ **−0.04173** or establishes cognitive deterioration. No denoising policy is selected by sign.'''),
C('''from pat16.pc_sampling import OUT as PCSAM, PROTOCOLS as SP, MODES as SM, DESIGNS as SD
from pat16.pc_sampling_figures import LABELS, MLABELS
sampling_design=json.loads((PCSAM/"design.json").read_text())
ss=pd.read_csv(PCSAM/"summary.csv")
sq=pd.read_csv(PCSAM/"sampling_qc.csv")
fm=pd.read_csv(PCSAM/"fc_metrics.csv")
nr=pd.read_csv(PCSAM/"network_ranges.csv")
fw=np.load(PCSAM/"frame_weights.npy",mmap_mode="r")
display(pd.read_csv(PCSAM/"feasibility.csv"))
print("Maximum bootstrap replay error:",sampling_design["bootstrap_max_absolute_replay_difference"])
display(Image(filename=str(RESULTS/"figures/pc_sampling_ranges.png")))'''),
M('''## Four sampling designs, three nuisance modes
**Bootstrap:** exact saved frequency-weighted draws, 146/150 rows including repeats. **Unit coverage:** same bootstrap draw's distinct frames once; it isolates multiplicities at identical coverage, and is not independent block subsampling. **Blocks, matched:** random circular origin on the original 180 positions; permute nonoverlapping 5/10/20-frame tiles without replacement, omit censored members, trim only the last tile to the bootstrap's exact distinct count. **Blocks, 80%:** same sampler, 116/120 retained frames. Censored gaps remain gaps; all temporal coordinates and derivatives remain original.

Median block-10 distinct counts are 93.5/96 pre/post, with frequency Kish counts 76.1/78.7 for the bootstrap. Kish counts describe concentration of frequency weights, **not independent BOLD sample size**. Matching distinct count does not match FD, temporal coverage, autocorrelation or physiology. Seeds/design/guards were fixed in `config/PC_SAMPLING.md` before computing contrasts. All 200 paired draws pass the existing safeguards in every design/block. Strict full-run bootstrap failure remains unchanged; earlier 68-frame duration/FD/run-half matching is in notebooks 07/10.

All modes use identical draws: frozen residuals; frozen PCs with regression refitted; or complete tissue preprocessing, PCA and regression refits. The same 500 whole-run SC bootstraps are crossed with 200 FC draws on the fixed 823 edges. The 100,000 cells are dependent. New support-reselected full-refit sensitivities are available at block 10.'''),
C('''def sampling_comparison(block=10,mode="full_refit"):
    v=ss[(ss.protocol.str.endswith("_block"+str(block)))&(ss.support=="fixed")&(ss["mode"]==mode)].copy()
    display(v[["protocol","mode","FC_draws","median","q025","q975","FC_share"]])
    display(sq[sq.block==block].groupby(["design","session"])[["rows","unique","frequency_Kish_count","mean_fd","first_half_unique"]].median())
    fig,ax=plt.subplots(figsize=(9,3));y=np.arange(len(v))
    ax.errorbar(v["median"],y,xerr=np.vstack([v["median"]-v.q025,v.q975-v["median"]]),fmt="o",capsize=4)
    ax.set_yticks(y,[LABELS[p.rsplit("_block",1)[0]] for p in v.protocol]);ax.invert_yaxis()
    ax.axvline(0,color="gray",ls="--");ax.axvline(-.0417277272,color="black",ls=":",label="Observed Δ")
    ax.set_xlabel("Post − pre coupling; conditional central range");ax.legend();plt.tight_layout();plt.show();plt.close(fig)
widgets.interact(sampling_comparison,block=widgets.Dropdown(options=[5,10,20],value=10),mode=widgets.Dropdown(options=list(SM),value="full_refit"))'''),
M('''## Inspect a saved draw and the exact multiplicity control
On the same original positions, bootstrap weights can exceed one. Unit coverage equals their indicator exactly. Matched-count block subsets have the same number of distinct frames but different positions. Each tile is used once; censoring and a trimmed last tile mean they are not contiguous short scans. Compare session-specific mean FD and thirds before interpreting coverage.'''),
C('''def inspect_sampling(block=10,draw=0):
    protocols=[d+"_block"+str(block) for d in SD]
    display(sq[sq.protocol.isin(protocols)&(sq.draw==draw)])
    display(fm[fm.protocol.isin(protocols)&(fm.draw==draw)&(fm["mode"]=="full_refit")][["protocol","session","unique","design_rank","distinct_rows_minus_rank","adjusted_PC_cos2_min","gap5_relative","cortical_mean_PC_removed","mean_fc_r"]])
    fig,axs=plt.subplots(2,1,figsize=(11,5),sharex=True)
    for t,ax in enumerate(axs):
        for design in SD:
            p=list(SP).index(design+"_block"+str(block));ax.step(np.arange(180)*2.4/60,fw[p,t,draw],where="mid",alpha=.7,label=LABELS[design])
        ax.set_ylabel(SESSIONS[t]+" weight")
    axs[0].legend(ncol=2,fontsize=8);axs[-1].set_xlabel("Original acquisition time (minutes)");plt.tight_layout();plt.show();plt.close(fig)
    a=fw[list(SP).index("bootstrap_block"+str(block)),:,draw]
    b=fw[list(SP).index("unit_coverage_block"+str(block)),:,draw]
    assert np.array_equal(a>0,b)
widgets.interact(inspect_sampling,block=widgets.Dropdown(options=[5,10,20],value=10),draw=widgets.IntSlider(min=0,max=199,value=0,continuous_update=False))'''),
M('''## Factorial training controls explain the shift
Refit all regression coefficients in every condition, on exactly the same weighted block-10 rows. Change tissue preprocessing only, PCA loadings only, or both. A separate temporal-only control freezes voxel scales/loadings. If background basis T is already in the final GLM, replacing tissue Z by Z−TA changes fixed-loading PC scores only within span(T), hence the fitted residuals are identical. PC signs and rotations within an unchanged span cannot explain a shift.

After projecting out the **same weighted motion/trend/DCT background**, the median minimum PC-span cosine² is 0.033/0.082 with bootstrap refits, versus 0.278/0.618 for unit coverage and 0.391/0.820 at 80% retention. This is not the earlier raw-PC-span metric. Baseline relative eigenvalue gaps (λ5−λ6)/λ5 are 0.015 pre-op and 0.046 post-op; a small gap is compatible with a sensitive five-PC boundary, but is not proof of a single PC swap. A larger gap after refitting does not certify stability relative to baseline.

PCA maximizes variance in the selected WM/CSF voxel data. More captured tissue variance does not guarantee removing the same cortical shared signal. Under bootstrap refitting, median cortical-mean variance removed relative to the same draw's background GLM drops from 0.899 to 0.600 pre-op and 0.902 to 0.806 post-op. This leaves more common cortical signal and raises mean FC, especially pre-op. Its neural/noise composition is unknown. Coupling ranks and network FC averages are different statistics and can move differently.'''),
C('''display(pd.read_csv(PCSAM/"mechanism_summary.csv"))
display(pd.read_csv(PCSAM/"mechanism_shifts.csv"))
display(Image(filename=str(RESULTS/"figures/pc_sampling_mechanism.png")))
display(Image(filename=str(RESULTS/"figures/pc_sampling_stability.png")))
mm=pd.read_csv(PCSAM/"mechanism_metrics.csv")
display(mm.groupby(["mode","session"])[["adjusted_PC_cos2_min","PCA5_fraction","gap5_relative","cortical_mean_PC_removed","mean_fc_r"]].median())
mechanism_fc=np.load(PCSAM/"mechanism_fc.npy",mmap_mode="r")
print("Temporal-only maximum |FC difference|:",np.max(np.abs(mechanism_fc[0]-mechanism_fc[1])))'''),
C('''def crossed_distribution(design="block_unique",mode="full_refit",block=10,support="fixed"):
    name=f"{design}_block{block}_{support}_{mode}.npz";path=PCSAM/name
    if not path.exists():
        display(Markdown("Support reselection here is saved only for the three new block-10 full-refit designs; the prior bootstrap reselection is in notebook 11."));return
    z=np.load(path);fig,ax=plt.subplots(figsize=(8,3))
    ax.hist(z["delta"].ravel(),bins=50,density=True,alpha=.7)
    ax.axvline(-.0417277272,color="black",ls=":",label="Observed Δ");ax.axvline(0,color="gray",ls="--")
    ax.set_xlabel("Post − pre coupling Δ (dependent crossed cells)");ax.set_ylabel("Empirical density");ax.legend();plt.tight_layout();plt.show();plt.close(fig)
widgets.interact(crossed_distribution,design=widgets.Dropdown(options=list(SD),value="block_unique"),mode=widgets.Dropdown(options=list(SM),value="full_refit"),block=widgets.Dropdown(options=[5,10,20],value=10),support=widgets.Dropdown(options=["fixed","reselected"]))'''),
M('''## Clinical interpretation and retained-fraction caveat
Visual FC's observed change remains −0.0045 Fisher z. Its full-refit median is −0.1993 under the bootstrap, −0.0458 at unit coverage, −0.0355 with matched-count blocks and −0.0143 with 80% retention. This displacement does not diagnose a visual deficit. The sensorimotor matched-count block range crosses zero, unlike the original full-refit bootstrap range. Dorsal-attention/default ranges remain negative in the new block-10 subsets, but these are exploratory, dependent summaries and do not localize clinical changes.

MOT slowing, RTI speeding, SSP span 6→7 and mixed SOC/RVP remain a mixed measured profile. Worry/loneliness, anxiety, emotion regulation and employment remain distinct unlocalized contextual observations. All 47 source fields, missingness and appendix dictionaries are preserved; no composite or clinical error bars are invented. Notebook [12](12_connectivity_cognition_context.ipynb) now compares the new network ranges beside each clinical field's candidate networks.

The 80% full-refit coupling range is [−0.0915,−0.0114] at block 10, but at 100% without replacement there is **no FC sampling spread** and the observed FC is recovered exactly (SC seed variability remains). Thus sampling-fraction-dependent ranges must not be substituted for calibrated full-run uncertainty or selected for a favorable sign. We retain full refitting for the estimator's sampling analysis, report the bootstrap and nonreplacement controls together, and do not claim one estimates population bias or neural truth.'''),
C('''display(Image(filename=str(RESULTS/"figures/pc_sampling_networks.png")))
display(nr[(nr.protocol.str.endswith("block10"))&(nr["mode"]=="full_refit")][["protocol","network","observed_delta","median","q025","q975"]])
links=pd.read_csv(PCSAM/"clinical_field_links.csv")
print("Unchanged clinical fields:",len(links),"; CANTAB:",(links.group=="CANTAB").sum())
display(pd.read_csv(PCSAM/"monte_carlo_halves.csv").query("mode == 'full_refit' and support == 'fixed'"))'''),
M('''## Reproduce and interrogate
`uv run python scripts/investigate_pc_sampling.py` verifies cached inputs or recomputes all designs, factorial controls and figures. `results/pc_sampling/design.json` fingerprints source, saved draws and outputs; `results/before_pc_sampling` preserves the preceding delivery.

**Exercises:** change block size and retained design; compare coefficient-only versus full refits at fixed coverage; explain temporal-only invariance; distinguish a small boundary eigengap from measured subspace overlap; explain why removing duplicates changes the estimator rather than repairs a conventional bootstrap; compare global coupling versus visual/sensorimotor FC; explain why the 80% negative range supplies no reliable-change test for MOT or QoL. Test the 100% sampler/fit identity with `tests/test_pc_sampling.py`. Independent repeat runs and physiological measurements are needed to resolve neural versus nuisance contributions.''')])



write('14_primary_fc_resampling.ipynb','# 14 | Recommended FC stability analysis and clinical interpretation\n\n**Question:** Which temporal perturbation matches our descriptive goal after diagnosing PC retraining?',[
M('''## Recommendation and target
Use **nonreplacement block subsets of 97 distinct eligible frames per session**, with complete WM/CSF temporal preprocessing, scaling, five-PC extraction and final regression refitted in every draw. Tile the original 180-frame axis into ten-frame blocks with a random circular origin; permute tiles once, omit censored frames and trim the final tile. Derivatives and DCT terms keep their original acquisition coordinates. Images, masks, motion and censor eligibility stay fixed.

Repeated bootstrap rows concentrate the empirical covariance used to learn PCs. Nonreplacement subsets remove that multiplicity mechanism; freezing full-pool training conditions on information outside the subset. A common count matches duration: floor(two-thirds × min(146,150))=97 frames, 3.88 minutes retained exposure, 58 algebraic residual directions after rank-39 regression. Two-thirds approximates previous 94/96 distinct coverage with a meaningful omitted fraction. This is a **post hoc engineering compromise**, saved in the policy before these new contrasts; no claim of optimality or calibrated coverage is made.

The target is **conditional stability of shorter-sample estimators**, not a full-run confidence interval or bias-corrected full-run outcome. Subsets overlap and the 500×200 grid cells are dependent. No probability of a biological direction, independent frame count, clinical effect or predictive model is estimated. Higher retention often narrows ranges because draws approach the fixed observed data; it cannot select a favourable sign.'''),
C('''PFC=RESULTS/"primary_fc"
policy=json.loads((PFC/"main_summary.json").read_text())
display(pd.Series(policy["policy"]))
display(pd.Series(policy["primary_joint_coupling"]))
display(Image(filename=str(RESULTS/"figures/primary_fc_joint.png")))
print("Observed full-run estimates are separate from subset medians and ranges.")'''),
M('''## Compare the specified sensitivities
The primary SC policy is unchanged: 823 fixed common-consensus edges with each session's all-run mean seed-rate weights, crossed with 500 whole-run SC bootstrap draws. Each of the nine new temporal designs has 200 draws per session; the index pairs independent session draws rather than physiological moments. The modes use identical subsets: frozen residuals, regression coefficients refitted with fixed PC scores, or complete nuisance refits.

Select common retained fraction and block length. `reselected` SC is available only for the primary full-refit FC branch. Previous bootstraps, unit-weight exact coverage, independent bootstrap-count-matched subsets, original session-specific 80% retention and 68-frame duration/FD/run-half matching remain in the catalog. The primary matches duration, **not motion**; matching sensitivities answer complementary conditional questions.'''),
C('''pfc_summary=pd.read_csv(PFC/"summary.csv")
def compare_primary(fraction="two_thirds",block=10,mode="full_refit",support="fixed"):
    protocol=f"{fraction}_block{block}"
    v=pfc_summary[(pfc_summary.protocol==protocol)&(pfc_summary["mode"]==mode)&(pfc_summary.support==support)]
    if v.empty:print("SC reselection is available for two_thirds/block10/full_refit only.");return
    display(v)
    grid=np.load(PFC/f"{protocol}_{support}_{mode}.npz")
    plt.hist(grid["delta"].ravel(),bins=55,density=True,alpha=.7)
    plt.axvline(policy["policy"]["observed_coupling"]["delta"],color="black",ls=":",label="Observed full run")
    plt.axvline(0,color="gray",ls="--");plt.xlabel("Post − pre SC–FC coupling");plt.ylabel("Grid density (dependent cells)");plt.legend();plt.show();plt.close()
widgets.interact(compare_primary,fraction=widgets.Dropdown(options=["half","two_thirds","four_fifths"],value="two_thirds"),block=widgets.Dropdown(options=[5,10,20],value=10),mode=widgets.Dropdown(options=["frozen","coefficients","full_refit"],value="full_refit"),support=widgets.Dropdown(options=["fixed","reselected"]))
catalog=pd.read_csv(PFC/"sensitivity_catalog.csv")
display(catalog.groupby(["family","role"]).size().rename("designs"))'''),
C('''from pat16.primary_fc import PROTOCOLS
frame_weights=np.load(PFC/"frame_weights.npy")
fit_qc=pd.read_csv(PFC/"fit_metrics.csv")
def inspect_subset(session="preop",draw=0,fraction="two_thirds",block=10):
    protocol=f"{fraction}_block{block}";w=frame_weights[PROTOCOLS.index(protocol),SESSIONS.index(session),draw]
    keep=np.load(RESULTS/session/"fc_primary_keep.npy");fd=np.load(RESULTS/session/"fd.npy")
    assert w.max()==1 and keep[w>0].all()
    print("Distinct frames:",int(w.sum()),"; exposure (min):",w.sum()*2.4/60)
    display(fit_qc[(fit_qc.protocol==protocol)&(fit_qc.session==session)&(fit_qc.draw==draw)])
    plt.plot(np.arange(180)*2.4,fd,label="FD");plt.scatter(np.where(w)[0]*2.4,fd[w>0],s=18,label="Selected distinct frames");plt.xlabel("Original acquisition time (s)");plt.ylabel("Approximate FD (mm)");plt.legend();plt.show();plt.close()
widgets.interact(inspect_subset,session=widgets.Dropdown(options=SESSIONS),draw=widgets.IntSlider(min=0,max=199),fraction=widgets.Dropdown(options=["half","two_thirds","four_fifths"],value="two_thirds"),block=widgets.Dropdown(options=[5,10,20],value=10))'''),
M('''## Clinical integration: observations remain measurements
Observed coupling Δ=−0.04173; recommended subset median −0.04694, range [−0.10261,+0.00832]. FC accounts for 98.33% of conditional crossed-grid variance, only under these perturbations. Sensorimotor range [−0.2064,−0.0170] coexists with slower MOT but faster RTI; a previous matched-count range crossed zero. Visual median −0.0209 is much closer to observed −0.0045 than the repeated-row median −0.1993. Visual, control and default primary ranges cross zero, while dorsal-attention is negative. Seven exploratory dependent network summaries do not identify a task-specific mechanism.

SSP 6→7, mixed SOC/RVP, anxiety 36→30, worry 54→61, loneliness 40→50 and employed→sick leave are different outcomes. Preserve all 47 source fields, units, missingness, RTI ceilings and unverified ERQ/education codes. No QoL composite, two-point correlation, reliable-change inference or causal surgery claim is supported. Appendices A1/A2 and notebook 04 retain the complete dictionaries.'''),
C('''clinical=pd.read_csv(PFC/"clinical_evidence_ledger.csv")
nr=pd.read_csv(PFC/"network_uncertainty.csv").query("role=='primary'")
display(Image(filename=str(RESULTS/"figures/primary_fc_network_changes.png")))
def clinical_primary(measure="rti_simple_rt_mean"):
    row=clinical[clinical.measure==measure].iloc[0]
    display(row.to_frame("Source and interpretation"))
    if pd.isna(row.candidate_networks):print("Unlocalized contextual evidence; no network effect or QoL score is inferred.");return
    display(nr[nr.network.isin(row.candidate_networks.split("; "))])
widgets.interact(clinical_primary,measure=widgets.Dropdown(options=clinical.measure.tolist(),value="rti_simple_rt_mean"))'''),
M('''## Reproduce and extend
`uv run python scripts/recommend_primary_fc.py` verifies hashes and reuses the checkpoint; `--force` recomputes all nine temporal designs and all paired nuisance modes. `scripts/plot_primary_fc.py` redraws figures. `scripts/run_all.py --notebooks --report` rebuilds the complete current package. See `docs/REPRODUCING.md` for clone/setup, runtime assets, external FSL and TeX requirements. The paper PDF is tracked by Git; publication of this locally prepared repository is deferred.

**Exercise:** compare 73, 97 and 116 frames with frozen and full refits. Explain why a narrower negative range at high retention does not validate a biological sign. Then inspect a contextual field with no network assignment and list what future measurements would be needed to test a clinical hypothesis.''')])
print('Built fifteen notebooks')
