# Primary FC resampling: paired fixed-duration block subsets with complete refits

Version: fixed-duration-nonreplacement-refit-v1. This is a post hoc methodological
recommendation informed by the preceding PC-training mechanism study, not an
independently validated or optimal sampling scheme. This document fixes the new
design before computing its contrasts. It leaves observed FC, primary anatomical
five-PC denoising, structural weighting/support and all clinical values unchanged.

## Primary design

- Work in the original 180-frame time coordinates, retaining the primary FD<=0.5
  eligibility masks (146 pre-op / 150 post-op frames).
- Set m = floor((2/3) * min(eligible pre, eligible post)) = 97 DISTINCT frames
  per session. Both have 3.88 minutes of retained exposure; acquisition span is
  longer and still contains censoring gaps. The fixed rule approximates the
  distinct coverage in the previous bootstrap (~94/96), retains a nontrivial
  omitted fraction, and avoids draw-dependent sample lengths. With rank 39 it
  leaves 58 algebraic residual directions, not 58 independent BOLD observations.
- Use 10-frame (24-s) original-time nonoverlapping circular tiles, a uniformly
  random origin, one random tile permutation, censoring inside tiles and a
  trimmed final tile. Sample without replacement; every retained row has weight
  one. This is a designed union of block fragments, not a contiguous scan,
  uniform individual-frame sampling or a classical calibrated subsampling CI.
- Generate 200 independently seeded paired subsets; within each session all
  cortical and WM/CSF signals share the same indices. Pre/post streams are
  independent, not paired original time points or matched physiological states.
- Refit tissue temporal projection, voxel scaling, all five WM/CSF PCs, nuisance
  design scaling and regression in EVERY draw. Motion derivatives and DCT
  coordinates stay on the original axis. Masks, corrected images and ROI data
  are fixed. Apply >=60 distinct frames, >=15 distinct-minus-rank and solver
  rank agreement; never redraw failures. Report ranges only with >=100 paired
  valid draws. Report eligibility counts even when all pass.
- Cross with the same 500 whole-run SC bootstraps on the fixed primary 823-edge
  common support, using each session's own repeat-mean seed-rate weights.
- Display the full-run observed point estimates separately from medians and
  central 2.5/97.5 percentiles of this conditional subset experiment. Do not
  replace the observed contrast by a subsample median, bias-correct it, use
  independent-cell errors, or label a subset range a full-run confidence interval.

## Rationale and limits

Without-replacement training avoids empirical covariance concentration from
repeated rows, the main loading-instability mechanism in the preceding study.
Refitting the complete estimator avoids conditioning on PCs or regression trained
on observations outside a subset. Common fixed m makes retained duration
comparable and decouples the primary design from the bootstrap's distinct-count
realizations. Two-thirds and 10 frames are transparent engineering compromises,
not selections for a desired coupling sign. No calibration or optimality follows
from them; the fraction is not asymptotically negligible and stationarity is
unverified. Matching duration does not match FD, temporal coverage, autocorrelation,
vigilance, vascular effects or independent information. Circular boundaries and
censored/trimmed blocks are explicit design limitations. Physiological specificity
and acquisition/model uncertainty remain unresolved.

## Preserved and new sensitivities

Before evaluating the new contrasts, also specify common counts floor(.5 * min)
=73 and floor(.8 * min)=116, and blocks 5/10/20 for each count. Use exactly paired
frozen residuals, fixed-PC coefficient refits and complete refits in each draw.
The primary is two_thirds_block10 + full_refit + fixed support. Reselect common
consensus as a sensitivity for that same protocol with full refits.

Use independent RNG streams 6160 + 100*block + 10*fraction_index + session_index,
with fraction order half, two_thirds, four_fifths and block order 5,10,20. Do not
reuse a favorable previous draw. Retain the original repeated-row bootstrap,
unit-coverage control, bootstrap-distinct-count-matched block subsets, original
session-specific 80% blocks, strict feasibility results, all six fully refitted
68-frame duration/FD/run-half matched analyses, SC support/weighting and nuisance
model alternatives as labelled sensitivities. None is ranked by clinical agreement.

## Clinical evidence

Main network uncertainty uses the new primary fully refitted FC subsets; SC
network uncertainty retains own-consensus support reselection under whole-run SC
bootstraps, with separate units and uncertainty sources. All 47 CANTAB/context
source values, missingness and caveats stay unchanged. Task-network links remain
broad hypotheses; context remains unlocalized. No two-point clinical association,
reliable-change test, cognitive/QoL composite or treatment effect is introduced.
