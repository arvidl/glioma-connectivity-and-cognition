# FSL diffusion correction

`b02b0.cnf` is copied unchanged from the installed FSL 6.0.7.19 distribution
(`etc/flirtsch/b02b0.cnf`). Its subsampling levels 2 and 1 support the 96×96×60
grid without padding. FSL binaries are external dependencies; they are not
redistributed. FSL has its own [license](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html).
This configuration retains the attribution and terms of that distribution.

Official implementation references:

- [topup guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/users_guide/index.html)
- [eddy guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/users_guide/index.html)

The pipeline generates acquisition rows from BIDS `PhaseEncodingDirection` and
`TotalReadoutTime` (seconds), never from the AP/PA filename alone. Each session
uses six AP b0s followed by two PA b0s. The first is raw AP volume 0, also eddy's
reference. All 102 AP volumes map to acquisition row 1. Raw AP data and original
FSL gradients enter eddy; downstream reconstruction uses eddy's rotated vectors.

Readout audit: supplied TRT is 0.0266003 s, while `EffectiveEchoSpacing` ×
(`PhaseEncodingLines` − 1) gives 0.03610038 s for the 96-line grid. The scanner
protocol is unavailable to resolve this legacy metadata discrepancy. We preserve
the supplied identical AP/PA value and use the SAME acquisition table in topup
and eddy. The shared time scaling largely cancels for displacement correction,
as explained in the [FSL FAQ](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/FAQ/index.html).
The physical Hz scale is therefore conditional on that choice; it is not an
independently calibrated fieldmap and must not be transferred to BOLD as-is.

Topup receives stacked voxel arrays with the AP header, as in `fslmerge`; no
interpolation or registration precedes the fit. AP/PA dimensions and voxel sizes
match. Small differing prescriptions (notably ~0.8° post-op) are approximated by
topup's rigid motion model. Original affine differences are saved in QC. Changes
above 1° stop this narrowly scoped workflow for explicit review. This is a
practical approximation and requires anatomical QC; it is not exact correction
for arbitrary oblique acquisition differences.

The stacked b0 file retains the AP int16 header. NIfTI storage scaling introduces
at most half an encoded intensity step (about 0.032 original intensity units in
this case), with no spatial interpolation. The volume-lineage test checks this
encoding bound rather than requiring bitwise intensity equality across storage
formats; original AP data enter eddy directly.

Eddy: CPU, eight threads, quadratic eddy-current model, five iterations,
outlier replacement, volume-to-volume motion, Jacobian-modulated resampling,
fixed seed 160. No slice-to-volume or susceptibility-by-motion model is fitted.
Topup uses its default single CPU thread; eddy uses eight. Tested FSL conda
package versions/builds and binary SHA256 hashes are in `fsl_environment.json`.
Command arrays, console logs, parameters, outlier maps, field (Hz), displacements,
Jacobians, corrected b0s and rotated gradients are retained. Nonpositive AP or PA Jacobians are excluded from the structural fitting mask. Internal AP/PA
agreement is a fit check; it is not independent evidence of anatomical accuracy.

The BOLD revision explicitly transfers this conditional Hz field into each BOLD
reference, scales using the BOLD readout and checks an alternative timing ratio.
See [BOLD correction](BOLD_SDC.md) for the composed sampling, validation and
cross-acquisition assumptions. This is separate from eddy and does not correct
BOLD slice timing or recover signal dropout.
