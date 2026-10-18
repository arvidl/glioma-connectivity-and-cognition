# Censoring reversal: conditional matching design

This post hoc experiment investigates the anatomical5 FD0.5 versus FD0.3
SC–FC coupling contrast. Primary derivatives and the denoising recommendation
remain unchanged. The report/provenance before matching are archived in
`results/censoring_investigation/before_matching` with primary-output hashes.

## Questions and controls

1. Ordered decomposition: primary residuals/primary frames → the same residuals
   restricted to strict frames → fixed primary PCs with regression coefficients
   refitted on strict frames → tissue temporal preprocessing/scaling/PCA and
   final regression all refitted on strict frames. Increments depend on this
   order and are computational diagnostics, not causal variance attribution.
2. Six sampling protocols: primary and strict eligible pools, each sampled for
   duration alone, duration + FD, and duration + FD + acquisition half.
   Each has 200 draws, 68 frames/session (2.72 minutes of retained exposure).
   All completely refitted draws use 39 design rank and 29 algebraic residual df.
   The original 180-frame time axis, high-pass basis and censor gaps are preserved.
   Retained exposure is not a contiguous acquisition or independent sample count.
3. Repeat every draw with frozen primary PCs AND regression coefficients, and
   with full retained-frame PCA/regression refitting. Frozen fits use training
   frames outside the selected draw and are sampling diagnostics, not independent
   short-run denoising pipelines. Their subset effective df is not inferred.
4. Coupling uses each session's fixed SC and own nonzero edge support as primary.
   Secondary controls use the intersection of nonzero SC edges (session-specific
   weights), then fixed preoperative SC weights on that same intersection. These
   alter the estimand. No tractography or anatomical reconstruction is rerun.

## Matching details and feasibility

FD-only matching uses a 0.015-mm absolute caliper per paired frame. The joint
temporal control adds equal frame counts in the first/second 90-frame acquisition
halves. Matching uses a random-cost maximum-cardinality bipartite assignment
(disallowed costs 10000, allowed independent U[0,1]); randomly select 68 valid
pairs without replacement. Frame FD, pool membership, run half and random seed
determine selection; no FC, SC or cognition enters it. This is not uniform
sampling of the original pools: it selects a shared motion-support population.
Matched frame pairs do not represent corresponding neural states.

A tighter 0.01-mm caliper admits 68 strict pairs with FD alone, but only 65 with
run halves, and 56 with thirds. At 0.015 mm the halves admit all 68, while thirds
admit only 60. We use 0.015 consistently in both motion protocols to keep the
same target frame count. This is a pragmatic feasibility choice, not a validated
biological tolerance. Half matching controls coarse temporal occupancy only;
it does not equate exact times, thirds, contiguous duration, fragmentation,
autocorrelation or vigilance. Raw retained FD, mean differences, standardized
mean difference, KS distance, Wasserstein distance, maximum paired FD gap,
acquisition-span, runs, adjacent-pair and occupancy counts are saved for audit.

Duration-only sampling is independent uniform selection without replacement in
each session's pool. The strict pre-op pool contains exactly 68 frames, so its
selected set (and completely refitted FC) is identical in all strict draws.
Seed 3160 + protocol index fixes independent streams. All selected masks are
stored, in session order preop/postop, in `matched_masks.npz`.

## Interpretation and reproduction

Median, empirical 2.5/97.5 percentiles and positive-draw fraction describe
conditional sampling-design sensitivity. They are not confidence intervals,
surgery-effect probabilities, p values, participant resamples or estimates of
test–retest reliability. Single-frame draws do not model temporal dependence.
FD matching does not guarantee absence of motion artifact or matching of spin
history, motion direction, physiology or other confounds. Algebraic df is not
an autocorrelation-adjusted effective sample size. These limitations constrain
inference even if every sampled contrast has the same sign.

`uv run python scripts/investigate_censoring.py` recomputes 200 draws for each
protocol from bundled corrected BOLD, existing tissue masks, ROI/motion signals
and SC. It needs no FSL rerun and uses one BLAS thread for small matrix fits.
Cached tissue signals retain the original boolean mask voxel order. Source/code
SHA256 hashes and design parameters invalidate stale experiment caches.
`--repeats` allows teaching runs; the delivered report uses 200.
`scripts/run_all.py` includes this experiment before notebook generation.
Notebook 07 selects a saved protocol, fitting mode and draw for live inspection.

The support, individual-edge and network-pair localization is in
`COUPLING_EDGE_DRIVERS.md` and notebook 08. Exact contributions reconstruct the
original and matched contrasts; reranked deletion has a different interpretation.
These diagnostics retain the existing primary and matching results.
