# Full-run nuisance training uncertainty and clinical evidence integration

This extension preserves `results/joint_sampling` and all primary point
estimates. It uses exactly the same saved original-time circular block indices
(5/10/20 frames), 200 draws per session/pool, and the same 500 bootstraps of the
20-run SC mean. No new favorable draws are selected, and no failed draw is
replaced. Fixed 823-edge and reselected 16/20 common consensus are both assessed.

## Paired nuisance conditions

1. Frozen: use the previously fitted session/pool residuals at sampled indices.
2. Coefficients: fix the session/pool's five anatomical WM/CSF PC scores, but
   refit all regression coefficients and design scaling on the sampled rows.
3. Full refit: refit tissue temporal projection, voxel means/SD, five PCs,
   nuisance-design scaling and regression on those same sampled rows.

All three use the same measured ROI signals, anatomical tissue masks, original
motion/derivative terms, original 180-frame time coordinates and DCT basis,
original FD censor eligibility and attempted exposure. Do not derive new motion
derivatives at artificial block joins, reindex the DCT basis, or select frames
according to FC/cognitive results. Spatial preprocessing and mask uncertainty
remain outside this experiment.

## Frequency weighting and feasibility

Let w_t be the multiplicity of original frame t. Temporal regression and nuisance
regression minimize sum_t w_t ||y_t-X_t beta||². Voxel centering/scaling and PCA
use the same multiplicities. Weighted unique-row PCA is algebraically equivalent
to explicitly repeating rows; tests compare it with the original PCA recipe.
FC is weighted correlation of refitted residuals, equivalent to correlation at
the sampled indices. Duplicated rows carry bootstrap frequency, not new measured
information. Save rank, solver rank, sampled row count, distinct count, count
minus rank, distinct count minus rank, condition number and residual variation.
Neither count-minus-rank is an estimate of independent BOLD degrees of freedom.

We conservatively apply the existing 60-frame/15-residual-direction feasibility
thresholds to DISTINCT original rows. This is a new, explicit bootstrap
feasibility safeguard, not a claim that all conventional bootstrap replicates
require 60 distinct observations. Both refits are attempted for every draw;
failed fits retain QC and NaN FC, and are not rescued by counting repeats as new
observations. This matters for strict pre-op, where the original fit used only
68 frames and duplicate-weighted resamples can approach interpolation.

Compare all three conditions on their identical paired eligible draws. Publish
central ranges only when at least 100 of the 200 planned paired draws qualify
(a prespecified descriptive minimum, not a calibration theorem). Retain sparse
eligible FC vectors/QC, but withhold a full-run refit uncertainty range when the
minimum is not met. Always show eligible/requested counts and the original
all-draw frozen results separately; conditioning on feasibility is not neutral
if rejections occur. No empirical sign criterion enters eligibility.

## Outputs and interpretation

Cross eligible FC draws with all SC draws. Store all FC vectors, per-session rho,
post-minus-pre delta, matched frozen/refitted shift grids, source/config hashes,
finite crossed-grid variance shares and Monte Carlo half checks. The paired
coefficient and PC-refitting shifts sum exactly to the full-minus-frozen shift.
They are computational changes, not causal physiological effects.

Clinical integration juxtaposes all 28 CANTAB and 19 psychosocial/context fields
with measured network summaries and their conditional uncertainty. Candidate
task/network links come from the pre-existing didactic hypotheses; they are not
individual localization or fitted predictions. Compute network SC means using
own consensus and all-run mean seed rates; network FC is mean Fisher-z within
network. Preserve raw units, missing values and undocumented coding. No
two-observation correlations, cognitive composites, cognitive bootstraps or
inferred reliable-change thresholds are generated. Contextual fields have no
assigned brain-network localization. No dedicated quality-of-life scale exists.

The original point estimates and all source assessments remain unchanged.
Clinical and connectivity ranges cannot be subtracted or correlated to claim a
mechanistic relationship. New assessments, clinical context and repeat imaging
would be needed to test any hypotheses. Sources are bundled and the preceding
delivery is archived under `results/before_joint_refit`.
