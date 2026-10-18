# PAT16: connectivity, cognition and lived experience

Latest extension: [notebook 13](notebooks/13_pc_sampling_mechanisms.ipynb)
diagnoses why nuisance-PC refits shift the bootstrap distributions, separating
preprocessing from PCA loading changes and comparing block subsampling without
replacement. [Notebook 12](notebooks/12_connectivity_cognition_context.ipynb)
adds these comparisons beside all CANTAB/context fields. Primary point estimates,
policies and clinical source values are unchanged. See the final section for
results, interpretation and reproduction commands.

## Primary SC recommendation (18 October 2026)

The main structural matrices, graphs, network means and teaching simulations
now use session-specific **16/20 seed-repeat consensus**, weighted by the mean
count across **all 20 runs including zeros**, per **10,000 attempted seeds**.
This retains 1,160/1,163 pre/post edges. For primary paired SC–FC coupling, use
the **823-edge intersection**, with each session's own mean-count weights, so
both correlations rank the same connections. Own-consensus coupling remains a
prominent sensitivity; session-specific consensus edges remain in structural
analyses. These questions deliberately use different supports.

The rule targets conditional seed stability and comparable edge populations;
it is a post hoc methodological recommendation, not calibrated anatomy or a
cutoff selected from coupling sign. The fixed seed denominator avoids the extra
compositional constraint from dividing by assigned-streamline totals; it still
reflects geometry, seeding-volume, tracking and endpoint biases, not axon counts.
See [the complete rationale](config/PRIMARY_SC.md).

Primary coupling is **0.1882 → 0.1465** (Δ **−0.04173**); strict full-refit FD0.3
is **0.1608 → 0.1265** (Δ **−0.03424**). Own-consensus contrasts are
−0.00257/+0.02363. All six matched-FC medians on common consensus are negative,
but five conditional central spreads cross zero. These results do not establish
biological decline, surgery effects or a cognitive/QoL mechanism.

New outputs are in `results/primary_sc`: policy/inputs, own/common matrices,
coupling/graph/network sensitivities, current edge/network rank accounting,
re-evaluated nuisance/BOLD/PC-count comparisons and matched-frame controls.
The earlier main delivery is archived under `results/before_primary_sc`, with
unchanged imaging, FC, original SC, seed runs and historical diagnostics protected
by hashes. Historical sections below describe the policy at the time of those
investigations. Notebooks 00–09 distinguish the current primary from those
historical sensitivities. CANTAB and contextual appendices remain unchanged.

Run `uv run python scripts/recommend_primary_sc.py` to rebuild the policy and
its sensitivity comparisons; `uv run python scripts/run_all.py --notebooks
--report` regenerates the full current delivery offline from bundled inputs.


A self-contained, exploratory paired case study prepared on 18 October 2026.
It contains **real pre/post T1, diffusion and resting BOLD**, newly computed
paired connectomes, all 28 supplied CANTAB measures, 19 psychosocial/context
fields, and explicitly hypothetical teaching simulations.

## Start here

```bash
cd /path/to/pat16-connectivity-case
uv sync --locked  # .python-version pins Python 3.12.14
uv run jupyter lab notebooks/
```

Use the Python kernel from this uv environment. Open notebooks in numbered order.
The delivered notebooks include executed static outputs. **Run All** in a live
Jupyter session activates the sliders/dropdowns; ordinary Plotly rotate/zoom
interaction also works in the saved HTML outputs. Plotly JavaScript is embedded,
so the analysis does not require a CDN or atlas download after installation.

Read `report/main.pdf` for the scientific case-study draft, and edit
`report/main.tex` / `report/references.bib` for the LaTeX/BibTeX sources.

| Notebook | What you can explore |
|---|---|
| 00 data, anatomy, design | Acquisition audit, lesion/atlas/EPI slice sliders, registration assumptions |
| 01 diffusion and SC | AP/PA correction audit and slice slider, motion/outliers, matched comparator, tensor/CSD, 3D tracts and graphs |
| 02 BOLD and FC | Static susceptibility-field transfer and slice QC, timing/PC sensitivities, motion/censoring, FD/GSR and residual-df guards |
| 03 longitudinal graphs | Matrix changes, threshold sliders, SC–FC coupling, network summaries |
| 04 cognition and context | Every CANTAB score, psychosocial context, selectable domain/network hypotheses |
| 05 simulations | Graph Fisher–KPP, hypothetical injury severity, OU dynamics and limiting cases |
| 06 nuisance models | PC cutoff slider, matched resampling/mask controls, T1 WM/CSF mask sliders, denoising matrices and tissue-threshold sensitivities |
| 07 censoring and coupling | Frame-selection/refitting decomposition, matched duration/FD/run halves, saved-draw slider, SC-support/reference controls |
| 08 edges and networks | Shared/session-specific SC, exact rank accounting, edge/network browsers, reranked deletions and matched support components |
| 09 SC reliability | Twenty new tracking runs/session, edge occurrence/count variability, count/consensus cutoff sliders and fixed-SC matched FC |
| 10 joint uncertainty | Whole-run SC bootstrap crossed with FC blocks and fully refitted duration/FD/half matching; fixed/reselected support; saved-cell and variance browsers |
| 11 nuisance refitting | Paired frozen/coefficient/full-PC refits, block/support browsers, distinct-frame feasibility and paired shifts |
| 12 integrated evidence | Every CANTAB/context field alongside candidate-network SC/FC changes, conditional ranges, sampling-design comparisons, raw units and missingness |
| 13 PC sampling mechanisms | Factorial preprocessing/loading controls, exact multiplicity comparison, nonreplacement block subsets, PC-span/QC and saved-draw browsers |

## What the data support

The cognitive profile is mixed. Reaction times are faster and spatial span rises,
while MOT latency/error increase; planning outcomes differ by difficulty.
Anxiety falls while worry/loneliness rise and employment becomes sick leave.
There is **no dedicated quality-of-life scale**, normative CANTAB score, reliable
change threshold, individual assessment date or treatment/medication history in
the bundled tables.

Paired SC and FC are newly reconstructed in a common Schaefer100 cortical atlas.
The original DK68 pre-op derivatives are retained as reference data, not used as
a postoperative substitute. Missing measurements never fall back to random data.

**DWI now uses AP/PA susceptibility and eddy-current correction.** FSL topup
fits six AP + two PA b0s per session; eddy_cpu jointly corrects susceptibility,
eddy currents and volume motion with slice-outlier replacement. Reconstruction
uses eddy's rotated gradients. The acquisition is **multi-shell** (6 b0, 16 at
700, 30 at 1200, 50 at 2800 s/mm²). Tensor FA/MD uses b≤1200; single-shell CSD
uses b0 + b=2800. This fixes the earlier mixed-shell CSD fit and incorrect protocol
description. The previous delivery is archived in `results/legacy_baseline`.
A same-model rigid-only comparison is in `results/uncorrected_matched` and
`results/correction_comparison.csv`.

**BOLD now receives static susceptibility correction.** The same-session AP/PA
field is rigidly transferred into the BOLD reference and scaled with BOLD readout
0.01464 s. Saved motion and signed PE displacement are composed to sample each
raw frame once with cubic interpolation and Jacobian modulation. Atlas/brain
sampling, nuisance PCs, FC and SC–FC coupling are recomputed; corrected SC and
motion/censoring are unchanged. The previous BOLD/FC pipeline is preserved under
`results/bold_uncorrected`. An alternative timing interpretation gives 0.761 times
the primary displacement (`results/bold_readout_alternative`). Notebook 02 and
`config/BOLD_SDC.md` explain the implementation, phantom checks and limitations.

**Interpret the connectivity as provisional.** DWI was acquired 9–10 minutes
after BOLD; shim compatibility and field stability cannot be verified. Readout
metadata are inconsistent, and timing sensitivity does not validate the field
transfer. Slice timing and motion-dependent susceptibility remain uncorrected;
signal dropout/neurovascular uncoupling cannot be recovered by this warp.
Atlas mapping is affine and longitudinal transport rigid; ACT/SIFT2 and validated
tissue-specific nuisance masks still require independent review. The post-op AP/PA prescription differs
by about 0.8°, approximated by topup's rigid model. See `config/README.md` for the
DWI audit and conditional Hz scale.

**Recommended primary denoising:** `anatomical5`, five PCs from combined,
session-specific T1 WM/CSF cores (PVE≥0.95), with temporal projection, scaling
and PCA fitted only on retained original indices. Final regression includes
24 motion terms and trend/DCT; FD>0.5 mm plus the next frame is censored;
no explicit GSR or low-pass. This targets anatomically defined nuisance tissue,
avoids training on censored frames and keeps the five-component budget
(146/150 frames, 107/111 residual df). It is a post hoc recommendation, not
preregistration or an independently validated optimum. The recipe and tradeoffs
are documented in [config/PRIMARY_DENOISING.md](config/PRIMARY_DENOISING.md).

At this historical denoising stage, mean FC was **0.087 → 0.046** and original seed-160 own-support coupling **0.210 → 0.189**. Current common-consensus coupling is **0.188 → 0.146**.
All main matrices, graphs, networks, conditional block-resampling intervals,
FC widgets and matched BOLD processing comparisons now use this recipe.
FD sensitivities refit PCA on their own retained samples. The original noncortical
five-PC method, base motion/DCT, raw-fit anatomical PCs, separate WM5+CSF5 and
tissue means remain sensitivities, together with PVE/count, GSR, filtering,
readout and graph-construction choices. Historical main FC/report/tables are
frozen under `results/before_primary_anatomical5`; structural matrices,
spatial preprocessing and source assessments are unchanged.

The historical postoperative sensitivity involves resampling, mask membership
and PC truncation: a cubic zero-field control reproduces much of the diagnostic
shift. With the anatomical recipe, the old/new postoperative BOLD package gap
is about 0.0024 (historically 0.1384), but about 90% of cortical-mean residual
variance is removed. Neither lower FC nor the smaller gap establishes neural
specificity. PVE-threshold sensitivity is modest; component count still matters. The full
anatomical 0–20-PC audit is exported as `results/primary_pc_sensitivity.csv`,
including adjacent-count results in notebook 06 and the report.
No causal surgery claim, reliable cognitive change or patient-specific prediction
is supported. Simulations remain teaching scenarios. See notebook 06 and
`config/NUISANCE_MODELS.md` for all retained models and diagnostics.
Notebook 04 and report Appendices A.1/A.2 now explain all 47 measures/fields,
units, direction, protocol ambiguities and missingness.

## Why the strict-censoring coupling reversal is unresolved

The original post−pre contrast is −0.0207 at FD0.5 and +0.0111 at FD0.3.
Restricting primary residuals to strict frames gives −0.0116; refitting regression
with primary PCs fixed gives −0.0027; refitting tissue PCA then crosses zero.
The pre-op estimate changes more during refitting. Frame selection alone does
not reverse the contrast.

New experiments use **200 draws per protocol, 68 frames/session**, with both
frozen primary fits and complete PCA/regression refits. Full-refit strict-pool
median contrasts are −0.0069 (duration), −0.0135 (duration+FD), and +0.0101
(duration+FD+run halves). Their central resampling ranges all span zero.
Motion matching uses a 0.015-mm absolute FD caliper; every matched draw has
absolute FD SMD below 0.070. Half matching alters which post-op frames enter
its small-sample nuisance fit. Equal duration and FD therefore do not settle
the direction or identify a biological mechanism.

With the primary fit frozen, all six protocol medians are negative. Restricting
SC to the **1071 shared nonzero edges** also removes the original reversal
(primary −0.0402; strict −0.0376), while fixing pre-op SC weights changes the
estimand again. This is an interaction among selection, nuisance fitting and SC
reference/support, not a reliable surgery-related coupling increase or decrease.
The existing anatomical5/FD0.5 primary remains fixed on methodological grounds.

Notebook 07 and the report show decomposition, matching/balance tables and
SC-reference controls. See [config/CENSORING_MATCHING.md](config/CENSORING_MATCHING.md)
for feasibility, seeds, frame masks and limitations. Retained exposure (2.72 min)
is not contiguous acquisition time or independent sample size. All matched
refits have 29 algebraic residual df; the empirical ranges are conditional
sampling-design spreads, **not confidence intervals or surgery-effect probabilities**.
These single-frame draws do not model serial dependence or repeated acquisitions.

## Which edges and networks account for the reversal?

The union of reconstructed SC contains **1071 shared, 444 pre-op-only and 488
post-op-only edges**. Exact edge rank products reconstruct Spearman coupling
with its ties. Their strict-minus-primary longitudinal shift is +0.03186:
pre-only edges contribute **+0.02437**, shared edges **+0.00833**, and post-only
edges **−0.00084**. These are signed accounting terms in the original pooled
ranks, not subgroup correlations or causal importance scores. Session-only
presence is not demonstrated anatomical tract loss/gain.

The largest positive network-pair terms are Default–SomMot (+0.01237),
Default–SalVentAttn (+0.00896) and Default–Limbic (+0.00644). Within-Default
(−0.00700) and several other pairs oppose the shift. Every one of 2003
single-edge deletions preserves the reversal. Reranked group deletion gives a
different account: removing Default-incident edges makes both contrasts
positive, while removing SalVentAttn-, Limbic- or Cont-incident edges makes both
negative. Removing post-only edges also removes the reversal, despite their
small original additive term, because all surviving ranks change.

All 6×200 saved matching draws are replayed and checked against the preceding
results. Shared-edge mean terms account for most of the change between strict
FD-only and FD+run-half matching; the pre-only term is fixed because all strict
pre-op samples are retained. The original and matched diagnostics therefore do
not identify one stable neural pathway or cognitive mechanism. Primary analyses
and preceding matched results are unchanged.

Notebook 08 includes full source atlas labels/IDs, measured FC and counts,
support/pair selectors, centroid edge schematics, deletion controls and matched
components. Straight centroid lines are not tracts; endpoint proximity does not
establish tumour/cavity traversal. See
[config/COUPLING_EDGE_DRIVERS.md](config/COUPLING_EDGE_DRIVERS.md) for the exact
identity, changed-rank caveat, complete CSV inventory and conditional controls.

## Conditional structural edge reliability

Notebook 09 adds **20 new tractography runs per session at 30,000 seeds/run**.
Only random seed locations vary; scans, CSD response/model, geometry, masks,
direction/stopping rules and 6-mm endpoint assignment stay fixed. Separate
seed-160 replays reproduce both original count matrices exactly. The same CSD
model is refitted once in float64 to recover the original tracking precision;
the primary float32 cache and all previous outputs remain unchanged.

Original singletons have median detection 55%/50% pre/post, versus 100%/100%
for original counts ≥10. Opposite-session rediscovery occurs for 322/444 original
pre-only and 342/488 post-only edges, but only 31/24 reach 80% occurrence in the
previously absent session. Edge 21–42 (the largest positive original reversal
term) occurs in just 3/20 pre-op and 2/20 post-op runs.

All 4,950 cortical pairs have per-run counts, occurrence and count SD/CV,
including zeros. Original shared/pre-only/post-only/absent labels remain fixed,
so opposite-session rediscovery can be inspected. The saved evidence includes
actual new seed positions and streamline endpoints/lengths, rather than a
bootstrap of the original tractogram. Count cutoffs are inclusive **1, 2, 3,
5, 10, 20** at the same 30k-seed budget. Consensus controls require occurrence
**0.5, 0.8, 0.95, 1.0**, weighting retained edges by mean counts per 30k seeds.

Coupling reranks every retained support. Own-session and common support are
separate analyses. All **400 Cartesian pre/post run combinations** are evaluated;
they reuse runs, so their empirical ranges/sign fractions are conditional seed
sensitivity, not confidence intervals or independent replications. All six
previous matched FC protocols are also replayed with full nuisance refitting
under five fixed SC references (original count ≥1/3/5; 80% own/common consensus).
This distinguishes varying SC at fixed FC from varying FC at fixed SC; joint
FC×seed uncertainty is quantified separately in notebook 10 under the recommended primary strategy. The preceding separate-source spreads must not be added.

Thresholding changes the edge population and ranks. Original own-support
contrasts become both positive at cutoff 2, both negative at 5/10, and both
positive at 20; common-support contrasts are negative across all six cutoffs.
With repeated SC at count ≥1, the original negative-primary/positive-strict
pattern occurs in 174/400 own-support combinations. An 80% common consensus
retains 823 edges and gives Δprimary −0.04173, Δstrict −0.03424. These are fixed-FC
point estimates; matched FC selection still matters. No cutoff is selected by sign. The original SC/anatomical5 FD0.5 fit remains the
historical reference; repeated/threshold/consensus results initially served as structural
sensitivity analyses. Fixed-image seeding measures conditional reconstruction
stability, not scan–rescan reliability, anatomical validity or tract loss/gain.
Systematic false connections can be stable. See `config/SC_RELIABILITY.md` and
the report for measured occurrence and coupling results.

## Historical frozen full-run joint uncertainty (retained sensitivity)

The primary rule and point estimates remain unchanged. The new experiment
crosses **500 bootstrap draws of the 20-run SC mean** with **200 paired FC draws**
per protocol. Whole-run SC sampling preserves edge covariance; FC resampling
moves all 100 ROI signals together. The main analysis fixes the 823-edge common
support. A sensitivity reapplies the same 16/20 consensus rule within each draw
and reranks SC/FC on its intersection.

Full-run FC resamples 10-frame circular blocks on the original temporal axis,
omits censored members inside each block, and retains the original exposure.
Blocks of 5/20 frames test the temporal scale. These draws freeze the fitted
nuisance residuals. The six saved 68-distinct-frame duration / FD / run-half
matched protocols instead **refit all nuisance steps**. Strict pre-op has exactly
68 eligible frames, so matched FC is constant there. The two branches answer
different conditional sampling questions.

For full-primary, fixed-support joint sampling, the median post-minus-pre
coupling is **−0.04050**, with conditional 2.5/97.5 percentiles
**[−0.08390, +0.01584]**. The SC-only anchor is [−0.04934, −0.03596].
FC accounts for **98.21%** of the finite crossed-grid variance, SC for 1.48%,
interaction for 0.30%. Strict FD-matched joint sampling remains negative in its
central range [−0.09073, −0.01323], whereas adding run-half matching gives
[−0.05720, +0.01066]. Support reselection gives a median 826 common edges and
changes the coupling population. Its strict FD-only upper tail changes sign across FC half ensembles; a small negative full-ensemble endpoint is fragile. Full-run block ranges cross zero under both
support modes at all three block lengths.

The 100,000 grid cells per design reuse 500/200 draws and are **not independent
replications**. Ranges and sign fractions are conditional computational
sensitivity, not calibrated clinical confidence intervals or surgical-effect
probabilities. Full-run sampling conditions on nuisance training; matching
conditions on its masks and short eligible pools. Acquisition, image noise,
registration, masks, diffusion model, haemodynamics and clinical measures are
not resampled. See `config/JOINT_SAMPLING.md` for all assumptions and seeds.

```sh
uv run python scripts/investigate_joint_sampling.py
uv run python scripts/run_all.py --notebooks --report
```

`results/joint_sampling/` includes every run index, block index, FC vector,
SC weight/support, matched mask, QC, 24 crossed grids, summaries, variance
components, Monte Carlo half checks and source/output hashes. The preceding
primary delivery is preserved in `results/before_joint_sampling/`.

## Files and provenance

- `data/bids/`: complete copied PAT16 BIDS subset, including source TSV/JSON,
  raw images, AP/PA gradients, lesion masks and reference TVB/rsHRF derivatives.
- `data/atlas/`: Schaefer100/7-network atlas, label table, matching FSL MNI152
  templates and attribution. Atlas tag and download origin are recorded.
- `data/source_context/`: source notices and extracted presentation text.
- `data/manifest.json`: SHA256, byte counts, input origins and package versions.
- `src/pat16/`: preprocessing, connectivity, clinical, network and viewer code.
- `results/{preop,postop}/correction/`: topup field, displacements, Jacobians, b0s, acquisition tables, exact commands/logs.
- `results/{preop,postop}/`: corrected DWI and BOLD (`func_sdc.nii.gz`), pull transforms, rotated
  gradients, FA/MD/CSD, tractograms, SC/FC matrices, ROI time series and QC.
- `results/{preop,postop}/bold_sdc/`: transferred field, rigid pull and registration audit in the session's `bold_sdc.json`.
- `results/bold_uncorrected/`: frozen previous BOLD pipeline and report; SHA256 preservation manifests.
- `results/bold_readout_alternative/`: alternative readout-ratio correction and recomputed FC.
- `results/bold_*.csv`: correction QC, fixed-SC FC/coupling comparison and nuisance-PC diagnostic.
- `results/nuisance_investigation/`: native FAST outputs, WM/CSF masks, alternative regressors/matrices, matched cubic control and preserved prior report/provenance.
- `results/censoring_investigation/`: source/code-hashed matched experiments, every selected mask, per-draw balance/coupling, tissue cache, ordered decomposition and archived preceding report/provenance.
- `results/coupling_edge_drivers/`: complete edge inventory, support/pair contributions, rank context, all single-edge and group deletion controls, matched per-edge arrays/summaries and preserved preceding delivery.
- `results/edge_driver_validation.json`: historical scientific checks and report QA for edge localization.
- `results/sc_reliability/`: new tracking counts, seeds/endpoints/QC, float64 CSD bridge, complete 4,950-edge reliability inventory, threshold/consensus coupling, all 400 repeat pairs and matched FC controls.
- `results/sc_reliability_validation.json`: current scientific contracts, notebook execution, PDF QA and preserved-input audit.
- `results/censoring_validation.json`: tests, executed notebooks, report verification and immutable-primary audit for this update.
- `results/nuisance_*.csv`: cutoff, resampling/mask factorial, denoising, tissue-composition, network and threshold/censoring comparisons.
- `results/*.csv`: all assessments, nodes/lesion exposure, metrics and sensitivities.
- `results/figures/`: publication-resolution PNG figures used by the report.
- `results/simulation_*`: explicitly simulated outputs.
- `results/notebook_execution.json`: completed cell/execution counts.
- `report/`: complete LaTeX project, generated numeric tables and compiled PDF.

All runtime inputs live **inside this folder**. You can move/copy it away from
its parent and the original repository. No symlinks or sibling-repository imports
are used. Recreate `.venv` with uv after moving the folder, since virtual
environments contain absolute launcher paths. A new machine needs uv/Python dependencies once; thereafter analysis
is offline. **Raw DWI recomputation additionally requires FSL topup/eddy_cpu**
(tested with FSL 6.0.7.19, binary build ID 2412.6-dirty). Set `FSLDIR` to your
installation (the local default is `~/fsl`); executables are found in
`$FSLDIR/share/fsl/bin` or PATH. uv does not install FSL. Saved-results notebooks
require only the locked Python environment. FSL has its own license; binaries
are not redistributed. `config/b02b0.cnf` is bundled and hashed.
New or deliberately forced T1 tissue segmentation additionally uses FSL FAST;
the saved segmentation and masks are bundled for offline notebook use.
A TeX installation is only needed for rebuilding the report PDF.

Raw input provenance: Aerts/Marinazzo BTC_preop OpenNeuro ds001226 v5.0.0 and
BTC_postop ds002080 v4.0.0, via the local
`glioma-brain-digital-twin-public-2026` release. Source notices are included.
The study is a new analysis; no clinical consent or ethics approval is newly
asserted here. Code is MIT; newly authored report/notebook prose and figures are
CC BY 4.0. Preserve third-party atlas/template and source notices.

## Recompute and verify

```bash
# Verify all bundled source inputs against the delivered manifest.
uv run python scripts/manifest.py --verify

# Rebuild tables/figures/notebooks; reuses the bundled repeated-tracking cache.
uv run python scripts/run_all.py

# Requires installed FSL: recompute raw scans, invalidating ALL stage checkpoints.
uv run python scripts/run_all.py --from-raw --force

# Recompute only BOLD correction/FC from saved topup fields; leave DWI/SC cached.
uv run python scripts/run_bold.py --force
uv run python scripts/run_all.py --notebooks --report

# Matched rigid-only DWI comparator and correction QC figures.
uv run python scripts/correction_comparison.py
uv run python scripts/correction_qc.py
uv run python scripts/bold_qc.py

# Nuisance-PC mechanism and anatomical WM/CSF comparison (cached FAST by default).
uv run python scripts/investigate_nuisance.py

# Censoring decomposition and 200 matched draws/protocol; bundled data, no FSL rerun.
uv run python scripts/investigate_censoring.py

# Edge/support/network accounting and replay of the same matching masks.
uv run python scripts/investigate_edge_drivers.py
uv run python scripts/investigate_sc_reliability.py

# Execute notebooks using this uv interpreter and project-local kernel config.
uv run python scripts/execute_notebooks.py

# Scientific-contract tests (endpoints, missingness, censoring, graph/simulation limits).
uv run pytest -q

# Generate tables/macros from CSV/JSON; then compile LaTeX and BibTeX.
uv run python scripts/report_tables.py
cd report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`run_all.py` deliberately regenerates notebook sources; run `execute_notebooks.py`
afterward to refresh their saved outputs. After changing scientific methods or
inputs, invalidate the affected stages (`run_bold.py --force` for BOLD-only changes,
`run_all.py --from-raw --force` for the full raw pipeline): stage completion files are **not** an automatic
dependency-tracking engine. `manifest.py` without `--verify` deliberately writes
a new input manifest; do not use it to conceal unexpected changes.

The notebook executor ignores headless widget comm mirroring because nbclient
0.11 can loop on large inline Plotly outputs. It executes real callbacks and saves
ordinary outputs; rerunning in Jupyter creates the live widget state normally.
No global kernelspec installation is necessary.

On Apple Silicon, all stages run on CPU. The code uses eight eddy CPU threads and six SimpleITK threads;
arrays are modest for 128 GB RAM. Inputs/results can be several GB including the
local Python environment and caches. Rebuilding the figures/report is much
faster than fitting/reconstructing from raw scans. CPU topup/eddy can take tens of
minutes per session; running the saved-result notebooks does not repeat it. Exact elapsed timings are recorded in
QC JSON; cross-platform floating-point and tractography differences are possible.

## Scientific upgrade path

Before interpreting biological longitudinal changes, verify cross-acquisition field/shim
compatibility and readout timing, and add lesion-aware nonlinear/surface
registration, a reviewed postoperative cavity
mask, expert review of the automatic tissue segmentation, physiological confounds and tractography/weighting sensitivity
with independent expert QC. Obtain assessment timing, treatments, retest norms,
validated QoL and repeated/cohort data before predictive or causal analysis.


## Full-run nuisance refits and clinical integration (18 October 2026)

The primary full-run uncertainty analysis now retrains **WM/CSF temporal
projection, voxel scaling, five PCs and nuisance regression in each FC draw**.
It reuses the same 200 original-time block draws and 500 whole-run SC bootstraps.
Frozen fits and fixed-PC/coefficient refits are exactly paired sensitivities.
The observed primary SC–FC contrast stays **−0.04173**. At ten-frame blocks and
fixed 823-edge support, full refitting gives median **−0.02069** and conditional
2.5/97.5-percentile range **[−0.09189, +0.05442]**, versus frozen **−0.04050
[−0.08390, +0.01584]**. These are computational sensitivity distributions, not
confidence intervals for surgery or cognition.

All primary pairs pass explicit guards of ≥60 distinct frames and ≥15 distinct
rows minus rank. **No strict-censoring pair passes in both sessions**, so strict
full-run refit ranges are withheld; all attempts and failed-draw QC are retained.
The distinct-frame guard is an explicit conservative choice, not a universal
bootstrap rule. Repeated rows do not increase measured information. Original
all-draw frozen and six fully refitted duration/FD/run-half matched designs
remain sensitivities in notebook 10. Details: `config/JOINT_REFIT.md`.

Notebook 11 compares the three nuisance conditions across blocks/supports and
inspects individual saved draws. Notebook 12 integrates all **28 CANTAB and
19 psychosocial/context fields**, retaining raw values, missingness and coding
limits, with within-network SC/FC uncertainty. Some retrained FC distributions
are far from their observed points; bootstrap centers do not replace measured
changes. Candidate task/network links are hypotheses; contextual fields have no
assigned network. The mixed clinical profile, absent retest norms and absent
dedicated QoL scale preclude a cognitive mechanism or quantified QoL conclusion.

```bash
uv run python scripts/investigate_joint_refit.py
uv run python scripts/integrate_clinical_evidence.py
uv run python scripts/run_all.py --notebooks --report
```

New arrays, exact paired coupling grids, feasibility and hashes are in
`results/joint_refit/`; the complete 47-field evidence ledger and network draws
are in `results/evidence_integration/`. Preceding mutable artifacts are archived
under `results/before_joint_refit/`; all earlier joint results, imaging, point
connectomes and clinical appendices remain protected and unchanged.


## Why PC refitting shifts the distributions (18 October 2026)

The shift is mainly associated with **relearning the PCA loading space under
repeated-frame weights**. On the exact ten-frame bootstrap draws, loading-only
refits shift the coupling contrast by a paired median **+0.02059**; tissue
preprocessing alone shifts it by **+0.00005**. Temporal projection alone with
fixed voxel scales/loadings leaves FC unchanged (maximum difference 3.20e-13),
because the final regression already includes the same temporal basis. Loading
and preprocessing increments depend on order and are not biological effects.

Bootstrap full refits change the background-adjusted PC spans substantially:
median minimum cosine² is **0.033/0.082** pre/post, versus **0.278/0.618** when
the same distinct frames each receive unit weight. Bootstrap median cortical-
mean variance removed falls from 89.9%/90.2% with fixed PCs to 60.0%/80.6% with
retrained PCs. Mean FC rises especially preoperatively. More captured WM/CSF
variance does not imply removing the same cortical shared signal; its neural
versus noise composition is unknown. Small baseline fifth/sixth eigenvalue gaps
are compatible with a sensitive truncation, without proving a single PC swap.

All designs below refit the full nuisance model and cross **200 FC draws with
the same 500 whole-run SC bootstraps**, on fixed 823-edge support. Every pair
passes the unchanged feasibility guards. The observed coupling contrast remains
**−0.04173**.

| Ten-frame sampling design | Median Δ | Conditional 2.5/97.5 percentiles |
|---|---:|---:|
| Original frequency-weighted bootstrap | −0.02069 | [−0.09189, +0.05442] |
| Exact bootstrap coverage, unit weights | −0.04366 | [−0.09162, +0.02834] |
| Independent nonoverlapping blocks, matched distinct count | −0.04001 | [−0.10186, +0.02461] |
| Independent nonoverlapping blocks, 80% retained | −0.04784 | [−0.09155, −0.01139] |

The full-minus-coefficient paired shift falls from **+0.02350** in the bootstrap
to **+0.00033** at unit coverage and **−0.00128** with matched-count blocks.
Unit coverage isolates multiplicity at identical positions; it is a deduplication
control, not independent block subsampling. The independent sampler randomizes a
circular tiling of original time into nonoverlapping blocks, omits censored
members and trims only the last tile to an exact target. Matching distinct
counts does not match motion, temporal coverage or physiology. Median distinct
counts are 93.5/96 pre/post; 80% retention uses 116/120. All comparisons also run
at block lengths 5 and 20; new block-10 full refits include support-reselection
sensitivities. Earlier strict and duration/FD/run-half controls are preserved.

These are **conditional sensitivity ranges**, not calibrated full-run confidence
intervals. At 100% without replacement FC sampling spread vanishes and the
observed fit is recovered exactly; SC seed variability can remain. Thus the
negative 80% range cannot resolve biological direction, justify choosing a
sampler by sign or support clinical deterioration. Empirical recentering does
not identify population bias. Keep complete refitting for the sampling estimator
and report bootstrap and nonreplacement controls together; the recommended
five-PC primary fit and SC support/weighting are unchanged.

Clinical interpretation now explicitly accounts for sampling design. Visual FC
has observed Δ −0.0045 Fisher z, versus medians −0.1993 (bootstrap), −0.0458 (unit
weights), −0.0355 (matched-count blocks) and −0.0143 (80% blocks). The large
bootstrap displacement is not evidence of a visual deficit. The sensorimotor
matched-count range crosses zero, unlike the bootstrap range. Dorsal-attention
and default ranges remain negative in these new block-10 subsets, but are
exploratory network summaries, not localized clinical change tests. MOT slowing,
RTI speeding, SSP span gain and mixed SOC/RVP remain a mixed profile. Anxiety,
worry, loneliness, emotion regulation and employment remain distinct unlocalized
context fields. No values, missingness, dictionary caveats or Appendices A1/A2
are changed; no cognitive/QoL composite or two-point association is calculated.

```bash
uv run python scripts/investigate_pc_sampling.py
uv run python scripts/run_all.py
uv run python scripts/execute_notebooks.py
uv run python scripts/report_tables.py
(cd report && latexmk -pdf main.tex)
uv run python -m pytest -q
uv run python scripts/validate_pc_sampling.py
uv run python scripts/record_analysis.py
```

`config/PC_SAMPLING.md` fixes the design before contrast computation.
`results/pc_sampling/` contains frame weights, tile origins/permutations, all
FC vectors, crossed coupling grids, factorial controls, network ranges, QC,
Monte Carlo halves, a 47-field clinical supplement and input/output hashes.
`results/before_pc_sampling/` archives the preceding mutable delivery; its
manifest also protects all earlier derived results and clinical inputs.
`results/pc_sampling_validation.json` audits the current delivery. Earlier
validation scripts/JSON describe their historical deliveries.
