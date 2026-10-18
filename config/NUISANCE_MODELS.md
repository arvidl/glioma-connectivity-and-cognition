# Anatomical nuisance comparison and PC-refitting investigation

This investigation retains the historical BOLD analysis alongside its anatomical
primary successor. Structural matrices are unchanged. All inputs, FAST tissue outputs, masks, regressors, FC
alternatives and numerical diagnostics are bundled. Notebook 06 runs offline
from these assets; no segmentation or raw image correction is repeated by its
widgets. Notebook 04 and report Appendices A.1/A.2 share the 47-field explanatory
dictionary in `assessment_annotations.json`, while preserving original TSV/JSON
values and descriptions.

## Recompute

```bash
uv run python scripts/investigate_nuisance.py
# Deliberately rerun FAST if T1/brain/lesion masks or segmentation settings change:
uv run python scripts/investigate_nuisance.py --segment-force
uv run python scripts/run_all.py --notebooks --report
uv run pytest -q
```

FSL FAST is required only for new tissue segmentation (tested with the existing
FSL 6.0.7.19 installation, build 2412.6-dirty); uv does not install FSL. Its exact
command, executable hash, input/mask hashes and class ordering are saved per
session in `results/nuisance_investigation/*/segmentation.json`. The FAST command
uses T1 input, 3 tissue classes, HMRF weight 0.1, 4 bias iterations, 20-mm bias
smoothing and bias-corrected/PVE outputs. FAST input retains only positive T1 intensities inside the session brain mask
and outside the lesion footprint plus 5 mm. Post-op
exclusion is the transported pre-op lesion, not a measured resection cavity.
Cached segmentation is explicit, not an automatic dependency graph. Full
`run_all.py --from-raw --force` also forces tissue segmentation; the matched
cubic control checks raw-image and saved-transform hashes before cache reuse.

WM/CSF masks require native fraction >=0.95, >2-mm T1 brain interior, and
>1-mm distance inside each tissue mask. A native core's nearest-neighbour
membership and linearly interpolated PVE >=0.95 must both pass after T1-to-BOLD
mapping. Apply BOLD brain/field support and the dilated lesion exclusion. CSF
includes ventricular and sulcal tissue. WM and CSF remain disjoint. The fractions
reported as composition/purity come from the same FAST model, not expert labels.

The six compared models all retain 24 motion terms, constant/linear/<0.01-Hz DCT,
the original censoring rule and fixed SC/nodes. `motion_dct` adds no tissue term;
`original5` reproduces historical noncortical PCs; `anatomical5_raw` changes
the mask but uses historical all-180-frame centering/scaling/PCA;
`anatomical5` uses combined WM/CSF and training on retained original indices
after constant/linear/DCT projection; `wm5_csf5` fits five PCs separately per
tissue; `wm_csf_means` uses two means. Scaling/PCA for retained-frame models
exclude censored samples; fitted spatial weights project all 180 frames.
Final regression is joint on retained samples, without concatenating gaps.
No explicit GSR is added. This custom CompCor-inspired implementation is not a
claim to reproduce fMRIPrep or to isolate physiological noise.

## Diagnostic controls and interpretation

PC sweeps and crossed PCA image/mask models keep corrected cortical time series
fixed. Historical linear, matched cubic zero-field and cubic corrected images
use identical saved motion. Five distributed-frame interior-mask checks confirm
order-1 composed sampling reproduces historical SimpleITK intensities (relative
RMS error <1e-6). The zero-field control sets PE shift to zero and Jacobian to
one and uses the corrected sampler; it is a PCA diagnostic, not another full
registration/ROI pipeline. Old/new masks, old mask restricted to field-valid
support and their intersection expose membership sensitivity. Crossed masks
share numeric voxel indices; unwarping changes physical tissue correspondence.

The postoperative sensitivity is an interaction between resampling, masks and
the five-PC truncation. Cortical-mean-associated directions enter the truncated
subspace differently. PC signs and within-subspace rotations cannot explain it;
old PC6 and new PC5 are not a one-to-one swapped pair. Moving the cutoff to ten
reduces the old/new gap but is not a validated choice of component number.

Partial R² is in-sample variance removed from the unweighted cortical ROI mean
after base motion/DCT regression. The anatomy-based five-PC model removes about
90% of this variance and reduces the old/new postoperative package gap. These
facts do not establish better neural measurement: shared neural fluctuations
may also be removed. Physiological recordings, manually validated tissue masks,
field/shim compatibility, retest data and reliable-change cognitive norms remain
unavailable. No model is selected to agree with cognition. A small old/new
processing gap is not an estimate of biological uncertainty.

Sensitivity CSVs cover PVE 0.90/0.95/0.99, 2/5/10 PCs, and FD 0.3 mm with
PCA refitting (only 29/57 residual df). The old report/provenance are preserved in
`results/nuisance_investigation/before_anatomical_comparison/`.

## Primary recommendation

`anatomical5` is now primary throughout the main analysis. See
[PRIMARY_DENOISING.md](PRIMARY_DENOISING.md) for the post hoc decision, frozen
recipe, caveats and archive paths. Other models remain sensitivity analyses.
