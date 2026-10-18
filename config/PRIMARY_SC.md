# Primary structural support and weighting for PAT16

This is a post hoc methodological recommendation after the seed-reliability
study. Prior FC results were available; the decision was not blinded or
preregistered. The rule is chosen for sampling stability and comparable populations,
without optimizing FC, coupling magnitude/sign, graphs or cognitive agreement.
No further image correction, CSD fit or tractography is needed for this change.

## Frozen rule

* Use the 20 new independent seed-location runs per session, 30,000 attempted
  seeds each (pre-op 17000–17019, post-op 18000–18019). Exclude seed-160 replay.
* An edge passes a session's consensus support if its count is >=1 in >=16/20
  runs (80%). Keep the arithmetic mean count over ALL 20 runs, including zeros.
* Weight = mean count / 30,000 * 10,000, i.e. assigned streamlines per 10,000
  attempted seeds. Do not condition the mean on detection, multiply by detection
  again, divide by retained total, or apply an additional count cutoff.
* SC matrices, graphs, network means and pre-op teaching simulations use each
  session's own consensus (1160 pre-op, 1163 post-op edges). This avoids removing
  337/340 reliably reconstructed session-specific edges from structural topology.
* Primary longitudinal SC–FC coupling uses the fixed intersection of those
  supports (823 edges), further restricted to finite FC nodes in BOTH sessions.
  Retain each session's own mean-count weights on this identical population.
  This estimates changing SC/FC rank association on consistently reconstructed
  connections. It does not estimate coupling of the entire changing topology.
  Own-consensus coupling is a prominent complementary sensitivity.
* For graph efficiency and clustering use one maximum weight across the two own
  consensus matrices. Retain requested AND achieved proportional density. Means
  over atlas network blocks include zeros for excluded edges and exclude the
  diagonal, so the denominator does not change when support changes.
* Spearman uses positive SC and measured FC with average tie ranks. No FC zero
  imputation, independence p values or per-session pre/post rank averaging.

## Why this choice

Original singleton edges have median detection 55%/50% and median count CV
1.12/1.13; original count>=10 edges have median detection 100%, but high-count
filtering discards potentially stable weaker paths and changes network coverage.
An occurrence rule targets observed seed repeatability directly. The 80% rule
permits four misses rather than requiring 20/20; it is a pragmatic, transparent
cutoff, not an optimized/validated boundary. Fixed ten-run split consensus
Jaccard is .865/.861 and selected edges have about .95 held-out occurrence.
The same repeats select and estimate weights: splitting is a diagnostic, not
independent validation of final estimates or proof of bias-free selection.

Mean counts reduce arbitrary single-run quantization and use all runs; including
zeros avoids conditioning weights on successful detection. A fixed attempted-
seed denominator avoids the extra compositional constraint from dividing by
each session's assigned-streamline total. It still measures sampling yield per
seed, not axons, fibre density or absolute biological connectivity. Seed-mask
size, geometry, tracking yield and endpoint/registration bias can differ between
sessions; fixed seed count does not remove those biases. No ACT/SIFT2 is claimed.

Common support is chosen for paired coupling because changing edge populations
confound a pre/post rank comparison. It deliberately excludes consensus-specific
connections; conditioning on both sessions may select a different population and
cannot test disconnection. Own-support structural results retain those edges.
This paired rule uses post-op data and is not a prospective pre-op predictor.

## Required sensitivity analyses and limits

Retain original seed-160 fractions/counts, own/common support, per-run count
cutoffs 1/2/3/5/10/20, unfiltered repeat means, consensus 50/80/95/100% at those
cutoffs, all 400 dependent repeat pairs, alternative BOLD/nuisance models and all
six saved 200-draw matched FC analyses. Recompute graph/network summaries and
coupling under alternative support/weights; rank-invariant scalar normalization
can still affect longitudinal weighted graph summaries. Historical diagnostics
keep original SC and are explicitly labelled as such. The previous main delivery
is protected in results/before_primary_sc; new outputs are in results/primary_sc.

Stable tracking can repeat false pathways. This is conditional seed sensitivity,
not scan–rescan reliability or anatomical validation. Choice of support, count
cutoff and nuisance/temporal sampling still changes the coupling direction. Do
not call the primary negative contrast evidence of surgical loss, cognitive
decline or a CANTAB/QoL mechanism; do not report empirical spreads as CIs.

Reproduce: `uv run python scripts/recommend_primary_sc.py`, followed by the main
analysis and notebook/report generators, or `uv run python scripts/run_all.py
--notebooks --report`. No network access is required with bundled checkpoints.
