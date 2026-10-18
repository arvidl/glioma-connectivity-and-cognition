# Conditional joint tractography and FC sampling uncertainty

The primary policy stays `repeat80-mean-seed-rate-common-coupling-v1`:
16/20 detection consensus, all-run mean counts including zeros per 10,000
attempted seeds; paired coupling on the common support with session weights.
This additional experiment is exploratory and follows inspection of prior FC.
No parameter is chosen to favor a coupling sign or cognitive narrative.

## Crossed resampling

* SC: 500 draws, each resampling **20 whole 30k-seed runs with replacement**
  independently within pre/post sessions (NumPy seeds 4160/4161). Recompute
  the mean, including zeros. This estimates sampling of the 20-run estimator,
  rather than replacing it with a noisy single tractogram. Whole-run sampling
  preserves edge covariance; no edgewise bootstrap or arbitrary pairing of run
  numbers across sessions is used. The original seed-160 run is excluded.
* Primary conditional-support analysis: keep the observed 823-edge common
  consensus fixed; vary mean weights. Selection sensitivity: independently
  bootstrap each session's runs, recalculate the inclusive 16/20 rule and use
  the intersection, then rerank both SC and FC on that draw's common support.
  This changes the ranked edge population and is not the identical estimand.
* FC: 200 draws per protocol, independent of SC draws. Six saved duration /
  duration+FD / duration+FD+run-half protocols from primary and strict eligible
  pools each retain 68 **distinct** frames/session. Refit tissue temporal
  projection, scaling, five WM/CSF PCs and nuisance regression on every mask,
  retaining the original 180-frame temporal basis and all gaps. Masks and FD
  pairing are reused exactly, not redrawn according to a coupling outcome.
  Motion matching uses the existing 0.015-mm caliper; run-half matching is coarse.
* Full primary (146/150) and strict (68/96) exposure: circular blocks of 10
  original frames (24 s), with 5 and 20 frames (12/48 s) as sensitivities.
  Draw a start uniformly on the original 180-frame axis, wrap modulo 180, omit
  censored members inside each block, repeat until the session's original
  retained count is reached and truncate. Resample all 100 ROIs together.
  Never form blocks from a concatenation of uncensored frames. These draws
  **freeze the session's fitted residuals**, including its censor-specific
  nuisance fit. They quantify temporal sampling conditional on that fit;
  they do not estimate PCA/regression training uncertainty. Repeated indices
  reduce the distinct-frame count, which is saved; do not assign the original
  residual df to a bootstrap draw or refit duplicated rows as independent data.
  Circular stationarity is an approximation, especially across run endpoints.
  The 5/10/20 choices are fixed sensitivities, not an optimized block length.

Cross all 500 SC draws with all 200 paired FC draws in each protocol and support
mode. Save per-session rho and post-minus-pre delta for all 100,000 cells.
SC-only varies weights/support with observed full-run FC. FC-only varies FC
with the observed fixed-support SC. Therefore SC-only is an observed-FC anchor,
not exactly the SC main effect of the crossed matched experiment. FC-only is
identical in the two support modes; it conditions on the point support.

## Summaries and interpretation

Report medians, 2.5/97.5 percentiles, SD and empirical fractions of positive
contrasts. A complete finite-grid variance decomposition uses centered row
(SC) and column (FC) means plus their orthogonal interaction, with population
variance (ddof=0). These terms sum exactly to total grid variance. The product
resampling measure is a computational assumption, not demonstrated physical
independence of MRI errors. The 100,000 cells reuse 500/200 draws: do not treat
them as independent samples, attach binomial standard errors to sign fractions,
or claim that their large count gives 100,000 independent replications.

These are **conditional resampling ranges**, not calibrated confidence intervals
for neural change, a surgical effect, or probabilities that cognition improved.
Short, autocorrelated, censored series, selected common support, only 20 empirical
tractograms, and a discontinuous 16/20 selection boundary limit calibration.
Matched draws are subsamples without replacement, not a temporal bootstrap.
Strict pre-op has exactly 68 eligible frames: its matched FC never varies;
matching uncertainty there is postoperative only. Equality of retained duration,
FD and coarse run half does not equate independent information or neural states.
Neither branch resamples imaging noise, acquisition, registration, masks,
diffusion model, endpoint assignment, haemodynamics, treatment or CANTAB.

Store sampled run indices, block indices, masks, all FC vectors, SC weights,
selected supports, QC, source/code/config SHA256, cross grids and half-ensemble
checks. Half checks indicate finite Monte Carlo sensitivity, not convergence
proof. The preceding delivery and unchanged-input hashes are archived in
`results/before_joint_sampling/manifest.json`. Execute with:

```sh
uv run python scripts/investigate_joint_sampling.py
uv run python scripts/run_all.py --notebooks --report
```
