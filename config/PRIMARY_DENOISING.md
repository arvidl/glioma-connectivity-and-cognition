# Primary denoising decision for PAT16 (18 October 2026)

Use `anatomical5`: five components from the combined, session-specific T1 WM/CSF
mask, with temporal preprocessing, voxel scaling and PCA trained only on retained
original frame indices. This is a **post hoc methodological recommendation** after
inspection of the previously reported sensitivity analyses, not a preregistered
choice or a patient-validated optimum. Freeze this recipe for both sessions; do
not choose a model by cognitive agreement, the lowest FC or highest SC–FC coupling.

## Rationale and tradeoffs

1. Anatomical specificity: FAST-derived high-purity WM/CSF cores target nuisance
   compartments more directly than an eroded noncortical EPI mask, which has mean
   model-estimated GM fraction about 0.19. The anatomical masks' internal target
   fraction exceeds 0.996. These fractions do not independently validate anatomy.
   Both native and BOLD-projected masks have been inspected at representative
   slices. Expert lesion/cavity-aware segmentation remains needed.
2. Retained-frame fitting keeps censored high-motion frames out of tissue PCA,
   temporal fitting and scaling. Low-frequency projection before PCA and joint
   final motion/DCT regression use original time indices throughout.
3. Five combined PCs retain the earlier component budget (residual df 107/111)
   while allowing more than two tissue-mean nuisance directions. CompCor and
   aCompCor studies motivate this approach, without proving five is optimal here.
   WM contributes more voxels than CSF; combined PCA does not balance tissues.
   Separate five-per-tissue regression remains a ten-PC sensitivity. A variance
   fraction rule would allow unequal session complexity and has not been adopted.
4. FD 0.5 mm plus the following frame retains 146/150 frames (5.84/6.00 min).
   FD 0.3 mm leaves 29/57 residual df; use it as a refitted sensitivity rather than
   the main comparison. Keep the minimum 60 frames/15 residual-df guard.
5. No explicit GSR or low-pass in the primary model. GSR changes shared cortical
   covariance, and a 0.01–0.1-Hz filter fails the preoperative df guard. Report
   both sensitivities, with abstention where ineligible. Absence of explicit GSR
   does not imply preservation of global neural signal: anatomical regression
   removes about 91%/90% of the cortical mean's variance beyond motion/DCT.
6. Internal checks support usability, not biological superiority: PVE 0.90–0.99
   gives mean FC 0.084–0.089 pre-op / 0.045–0.048 post-op; the matched anatomical
   old/new postoperative BOLD package gap is about 0.0024. PC count still matters
   (full 0–20 sweep). Adjacent postoperative 4/5/6-PC means are
   0.0477/0.0456/0.0449; preoperative means 0.0864/0.0872/0.0721. These do
   not show the large historical postoperative cutoff jump, but preoperative
   count dependence remains appreciable; smaller processing dependence is not neural validation. One patient
   cannot support a cohort QC–FC benchmark or establish an optimal model.

## Frozen parameters and lineage

- Static susceptibility-corrected BOLD; same saved motion and raw cortical ROI
  series, lesion footprint exclusion, 100 common Schaefer nodes and corrected SC.
- T1 FAST three classes; existing brain mask; lesion footprint plus 5-mm exclusion;
  PVE >=0.95; >2-mm brain interior; >1-mm native erosion; NN core and linearly
  sampled PVE must both pass; valid BOLD field/FOV/Jacobian support.
- Combined WM/CSF voxel normalization and PCA after constant/linear/<0.01-Hz DCT
  projection; all coefficients fit on the retained frame set. Five PCs total.
- Drop four initial volumes; FD >0.5 mm plus next frame excluded; 50-mm radius.
- Final single joint fit: intercept, linear trend, 24 motion terms, five PCs and
  <0.01-Hz DCT terms; correlation on retained residuals, no explicit GSR/low-pass.
- FD sensitivities refit PCA on their own keep set. The GSR sensitivity uses the
  whole valid brain outside the footprint; the notebook uses this same signal.
- Primary FC, residuals, QC, graph metrics, network block resampling and SC–FC
  coupling all derive from this fit. `fc_qc.json` records `denoising_version`,
  `denoising_model`, input hashes, mask sizes and design df. Historical
  `nuisance_pcs.npy` retains the original noncortical PCs; the primary uses
  `primary_nuisance_pcs.npy`.

All six nuisance models, PVE/count variants, FD 0.3/1.0, whole-brain GSR, eligible
band-pass, readout alternatives and historical PC swap/resampling diagnostics
remain available. Every BOLD package in `bold_comparison.csv` now uses the SAME
anatomical primary recipe. The immutable rigid-only delivery stays under
`results/bold_uncorrected`; its newly matched anatomical derivatives are under
`results/nuisance_investigation/{session}/uncorrected`.
The preceding main matrices/tables/report are frozen with hashes under
`results/before_primary_anatomical5`; SC and spatial preprocessing are unchanged.

## Supporting primary literature

- Behzadi et al. (2007), CompCor, doi:10.1016/j.neuroimage.2007.04.042.
- Muschelli et al. (2014), aCompCor versus tissue means and motion artifacts,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4043948/ . Results in a different
  population motivate the method, not a PAT16 efficacy claim.
- Ciric et al. (2017), confound-strategy benchmarks and tradeoffs,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5483393/ . Cohort metrics cannot be
  estimated from these two scans; no universally best strategy is inferred.

The censoring reversal is investigated separately in `CENSORING_MATCHING.md`
and notebook 07. Matched duration/FD/run-half refits and SC-reference controls
leave the longitudinal coupling direction unresolved; they do not change this
primary recipe or justify selecting a threshold from the resulting sign.
