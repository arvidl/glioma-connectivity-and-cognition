# Conditional reliability of low-count/session-specific SC edges

This descriptive sensitivity analysis keeps the anatomical5 FD0.5 primary model,
FD0.3 strict fit, corrected DWI/BOLD, CSD response, geometry, seed/stop masks,
tracking settings and 6-mm endpoint assignment unchanged. It never overwrites
primary SC/FC or the preceding edge/matched analyses. The previous delivery is
archived in `results/sc_reliability/before_reliability` with immutable input hashes.

## Independent repeated tracking at a fixed sampling budget

For each session, 20 new random seed-location runs use 30,000 seeds each;
pre-op RNG seeds are 17000–17019 and post-op 18000–18019. They are distinct
across sessions and exclude the original seed 160. Tracking directions remain
deterministic: CSD SH6, default sphere, 35° maximum angle, PMF cutoff .1,
FA>.25 seeding, FA .1 stopping, .75-mm steps, max_cross=1, maxlen=400,
20–250-mm retained lengths and 6-mm nearest-atlas endpoint assignment.
We save seed locations, endpoint pairs/distances/lengths, counts and run QC.
These are newly generated streamlines, not resamples of an existing tractogram.

The original tracker used float64 coefficients but saved a float32 cache.
Refitting the same high-shell CSD model/response once recovers float64 tracking
coefficients. A separate seed-160 replay must reproduce the original count
matrix exactly before new repeats run. All original files remain unchanged.
No new acquisition, image registration, FSL correction or diffusion noise
bootstrap is performed. Repeated seeding probes only conditional sampling
stability, not scan–rescan reliability, anatomical validity or tract survival.

## Edge occurrence, count variability and support

All 4,950 upper-triangle cortical edges are evaluated, including edges absent
from both original runs. Detection at threshold k is count>=k in each individual
30,000-seed run, for k=1,2,3,5,10,20. Occurrence is hits/20 (resolution .05).
Count SD/CV include zero-count runs; all-zero means have undefined CV (NaN) and are excluded from median CV. Original shared/pre-only/post-only/absent
support labels are fixed, allowing opposite-session rediscovery to be measured.
Pairwise within-session support Jaccard and full-4,950-edge count-rank agreement
are exported. The fixed first/last-ten split compares 80% consensus supports
and held-out occurrence; it does not validate anatomy or estimate scan error.

An 80% consensus class is both/pre-only/post-only/neither according to which
sessions reach occurrence>=.8 at k=1. "Pre-only consensus" allows sub-.8
postoperative detection and must not be read as verified absence. We separately
flag occurrence>=.8 in one session and zero hits in the other; 0/20 still does
not demonstrate anatomical absence. Original reversal terms are cross-tabulated
by original and repeat consensus classes with exact signed accounting.

## SC–FC coupling and count-threshold controls

For original seed-160 SC, each inclusive count cutoff reranks all retained
positive-SC edges. Own support thresholds sessions separately; common support
uses the intersection at that cutoff with each session's own count weights.
Measured FC is unchanged; SC-absent FC values are excluded, never imputed.
The original primary and strict BOLD fits are used, saving both post-minus-pre
contrasts and R=strict contrast−primary contrast. Thresholding changes support,
ranks and the estimand; it is not simple subtraction of low-edge contributions.

For repeated SC, all 20×20 Cartesian pre/post run combinations are evaluated
under own/common support and all six count thresholds. Combinations reuse runs
and are dependent. Their empirical 2.5/50/97.5 percentiles and sign fractions
summarize conditional seed sensitivity, not confidence intervals, p values,
treatment probabilities or independent replication. There is no arbitrary
one-to-one pairing of distinct-session RNG IDs.

Consensus sensitivity uses occurrence fractions .5,.8,.95,1 at each count cutoff,
with mean counts per 30,000 seeds as weights. Cutoff applies within each run,
not to the sum over 600,000 seeds. Consensus own/common supports are distinct
from averaging correlations or validating a reconstructed pathway. No threshold
or occurrence fraction is selected from a preferred coupling sign.
All global ranks/normalizers are recomputed. Exact reversal terms are exported
by original support, repeat-consensus class and network pair for every fixed
threshold/consensus reference.

All 6×200 previous selected-frame masks are also replayed with complete nuisance
refits, holding five SC references fixed: original own SC at k=1,3,5 and 80%
consensus at k=1 with own/common supports. The original k=1 controls must exactly
reproduce the preceding matched FC/SC results. This separates frame-selection
sensitivity from fixed SC-reference sensitivity; joint FC×seed uncertainty is
not estimated. Historical frozen-fit analyses remain available unchanged.

## Reproduce

`uv run python scripts/investigate_sc_reliability.py` computes/reuses tracking
checkpoints, analyses and figures offline; `--tracking-only` stops after tracking.
The bundled counts/endpoint/seed/float64 CSD files support local reanalysis.
Use notebook 09 for interactive count/consensus cutoffs and per-edge inspection.
Source/input hashes protect each run design and all saved original outputs.
