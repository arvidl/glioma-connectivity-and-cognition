# Which edges and networks account for the censoring reversal?

This post hoc diagnostic uses the existing anatomical5 FD0.5 primary and FD0.3
strict fits, plus the two saved intermediate stages and all previous matched
draws. It changes no denoising, SC, primary FC or preceding matched results.
The preceding report, source, validation and provenance are archived under
`results/coupling_edge_drivers/before_edge_analysis` with immutable input hashes.

## The quantity being localized

Let delta(a) = rho(post,a) − rho(pre,a). The reversal shift is
R = delta(strict) − delta(primary), not a neural or surgical effect.
The original own-SC values are −0.0207203, +0.0111444 and R = +0.0318647.
The upper-triangle SC union contains 2003 edges: 1071 shared, 444 pre-op-only
and 488 post-op-only. These names denote streamline reconstruction presence,
not demonstrated tract loss or gain. Edges absent in both sessions are omitted.

## Exact additive rank accounting

For each session and fit, rank positive SC counts and the corresponding FC
values over that session's complete nonzero-SC support, using average ranks
for ties. Counts and log1p(fraction) have identical ranks. If x and y are
centered ranks, c_e = x_e*y_e / sqrt(sum(x²)*sum(y²)), so sum(c_e) = Spearman rho.
For accounting only, an absent edge's contribution is zero; its FC is never
replaced by zero or included in the ranks. The full union-edge reversal term is
r_e = (c_post,strict − c_pre,strict) − (c_post,primary − c_pre,primary).
Sum by support or unordered atlas network pair to recover R exactly. An edge
may have either sign; positive/negative terms can strongly cancel.

The rank universe and normalizer belong to the original pooled support.
Subgroup-only Spearman correlations are separately recomputed and do not sum
to rho. The shared-edge contribution in pooled ranks is therefore distinct
from coupling computed after excluding all session-specific SC edges.
Within/between-group covariance terms use the same global ranks and denominator:
sum((x−mean_g(x))*(y−mean_g(y)))/D + n_g*mean_g(x)*mean_g(y)/D.
They expose group-location effects without claiming a causal decomposition.

## Deletion and localization checks

Recompute all ranks and normalizers after deleting each support class, every
unordered network pair, and all edges incident on each of seven networks.
Network-incident sets overlap and their deletion effects must not be summed.
Also recompute every single union-edge deletion and cumulative deletion of
1/5/10/25/50/100 positive, negative or largest-absolute r_e edges. These ranked
sets were chosen from the observed sensitivity and are descriptive post hoc
stress tests, not validation samples or independently important neural edges.

Global-rank contribution and deletion effects answer different questions.
Deleting a group changes all surviving edge ranks and normalizers, so removing
a group with a small additive R term may have a large effect on delta's sign.
No edgewise p values, independent-edge confidence intervals or cognitive
localization are inferred. Short/low-count paths and single-session presence
are especially conditional on tractography and the endpoint rule.

Distances are straight-line pre-op T1 RAS atlas-centroid distances, not streamline
lengths. Proximity means either atlas endpoint is within 10 mm of the transported
preoperative footprint; it does not measure whether a tract traverses tumour or
the postoperative cavity. Atlas network labels are not patient-validated maps.
Notebook edge plots are straight centroid schematics, explicitly not tracts.

## Conditional matched controls

Replay the exact 6 × 200 saved masks with frozen and fully refitted nuisance
models. Every edge-contribution contrast is checked against the previous
matched coupling table. Store per-edge post−pre contrasts for each draw; their
support/pair sums reproduce that draw's global contrast. The additional
`refit_effect` is the refitted contrast minus the frozen-fit contrast on the
SAME selected frames. It is a fitting diagnostic, not the original full-sample
strict-minus-primary R. Do not pair unrelated primary/strict draws and claim a
new treatment or censor-threshold effect.

Matched group means sum to the mean total for disjoint supports/pairs. Marginal
group medians and quantiles generally do not sum. Spreads and sign fractions
are conditional on the two fixed acquisitions and selection designs, not
confidence intervals, effect probabilities or test–retest reliability. The
previous caveats about serial dependence, frame-count/FD balance and run-half
matching remain in `CENSORING_MATCHING.md`.

## Reproduce and inspect

`uv run python scripts/investigate_edge_drivers.py` uses bundled matrices, masks,
tissue time series and fixed original time axes; no imaging/FSL stage is rerun.
Source hashes invalidate the new experiment cache. Notebook 08 offers support,
network-pair and edge selection, decomposition and deletion tables, and matched
controls. CSV inventories retain source atlas labels/IDs, counts, measured FC,
FC ranks and contributions. `matched_edge_contrasts.npz` uses keys
`protocol__mode`, in the row order of `edges.csv`; each is draws × union edges.
All tables/figures are regenerated through `scripts/run_all.py`.
