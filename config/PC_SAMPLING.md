# Nuisance-PC sampling mechanism and block subsampling

The preceding full-run refit analysis and all observed SC/FC/clinical inputs
are preserved. This extension diagnoses an empirical distribution shift, not
population bias or neural signal specificity. The five-PC primary denoising
policy, common 823-edge SC support and 500 whole-run SC bootstraps stay fixed.
No parameter is chosen according to coupling sign or agreement with cognition.

## Paired designs, 200 draws at blocks 5/10/20

- `bootstrap`: replay the preceding frequency-weighted frozen/coefficient/full
  refits exactly, including the saved original-time circular blocks.
- `unit_coverage`: use exactly the distinct frames of each bootstrap draw once.
  This isolates multiplicity at fixed coverage; it is a deduplication control,
  not an independent uniform or block-without-replacement sample.
- `block_unique`: sample a random circular tiling of the original 180 positions
  into nonoverlapping blocks, permute tiles without replacement, omit censored
  frames inside each tile, and stop at the exact bootstrap distinct-frame count
  for that session/draw. Trim only the final tile if necessary. No frame repeats.
  This matches distinct duration, while changing temporal coverage; it does not
  match FD, physiology or independent information.
- `block80`: same block sampler, retain floor(0.8 * eligible primary frames),
  i.e. 116/120 frames. This tests higher retained fraction, not a calibrated
  confidence interval for the original full-run statistic. At 100% the sampler
  must recover the observed full-run fit exactly.

Circular tile origin is uniform over 0..179. All block lengths divide 180;
nondividing lengths are rejected. Original derivatives and temporal basis are
retained. Every ROI uses the same draw. Frozen, fixed-PC coefficient refits and
complete PC/regression refits use identical masks within each design. All use
the preceding distinct-frame/rank guards (60 unique, 15 unique-minus-rank;
minimum 100 paired draws for a range); failures are saved, never redrawn.
Pre/post RNG streams are separate, fixed from 5160 + 100*block + 10*design +
session index. Strict bootstrap feasibility is not rescued or redefined;
strict fully refitted matching remains an earlier separate sensitivity.

Cross every eligible FC draw with all 500 SC bootstraps on fixed support.
For block 10 also evaluate reselected common consensus for the three new full-
refit designs; previous bootstrap reselection remains available. Cells are
computationally dependent. Subsample spreads shrink with sampling fraction and
must not be relabeled full-run confidence intervals or chosen by their sign.

## Mechanism controls on the original primary block-10 bootstrap

Use a 2x2 comparison of tissue temporal-projection/voxel-scaling training
(baseline vs resample) and PCA spatial-loading training (baseline vs resample),
with all regression coefficients refitted in every condition. This separates
preprocessing-only and loading-only changes, with full refitting as the joint
condition. The ordering is diagnostic, not a unique causal decomposition.
Weighted recentering before loading estimation changes only the intercept part.
A fifth control changes tissue temporal projection alone while freezing voxel
SD and loadings: its score change lies in the existing trend/intercept/DCT
nuisance span, so FC should be invariant up to floating-point error. Varying
voxels must match the baseline; otherwise the diagnostic stops explicitly.

Save PCA eigenvalues around the 5/6 cutoff, variance fractions, weighted
frequency concentration (Kish count, not independent BOLD information), design
rank/condition and PC subspace overlap AFTER projecting out the common
motion/trend/DCT background on the same weighted rows. PC sign/rotation does
not change regression. Save cortical/global and within-network mean residual
variance after PCs relative to the same draw's background-only regression,
with shared-signal removal explicitly agnostic about neural vs nuisance signal.
Compare pre/post changes separately; global FC means, network Fisher-z means
and coupling on common SC edges are different estimands.

## Clinical interpretation

Retain all 47 field values, units, missingness and dictionary caveats unchanged.
Export a new clinical evidence supplement with network ranges under every
sampling design, emphasizing agreement and disagreement across controls.
No clinical composites, two-point correlations, reliable-change tests,
individual localization or quantified QoL effects are created. Bootstrap or
subsample medians do not replace observed clinical/connectivity changes.
