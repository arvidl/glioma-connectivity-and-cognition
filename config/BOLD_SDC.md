# Static AP/PA field transfer to BOLD

Implementation: `src/pat16/bold.py`, method `static-topup-transfer-v1`.

The same-session topup Hz field and corrected b0 reference share the AP-volume-0
reference used by eddy. They are rigidly registered to the BOLD early-run mean:
one initial mutual-information fit, then two fixed field-transfer/unwarp/refit
rounds. The fixed fitting mask excludes the dilated lesion footprint. Scalar Hz
values are linearly resampled through the final pull transform into BOLD geometry.

BIDS BOLD PE is `j-`; supplied total readout is 0.01464 s. In voxel coordinates,
`s(x) = -f_Hz(x) * TRT` and raw-volume pull coordinates are
`y_v(x) = M_v(x) + e_j * s(x)`. Thus saved rigid motion maps output coordinates
into a raw frame, then the displacement is added along that raw PE voxel axis.
There is one cubic interpolation from each raw frame and multiplication by the
exact determinant `det(B_v) * (1 + grad(s)^T inv(B_v) e_j)`. The reference/static
field is not a model of susceptibility changes with motion.

Positive finite Jacobians, field-brain support and in-bounds coordinates across
all 180 frames define a valid mask for ROI and nuisance sampling. Corrected means
are registered to T1 again. The original motion estimates and censoring remain
fixed; SC is unchanged. Slice timing is not corrected. Old BOLD/FC is frozen in
`results/bold_uncorrected` with hashes. Its interpolation was linear, and spatial
masks/PCs are refitted in the new pipeline, so this is a package comparison.

Readout audit:

| Acquisition | Supplied TRT (s) | EES × (phase lines − 1) (s) |
|---|---:|---:|
| DWI | 0.0266003 | 0.03610038 |
| BOLD | 0.01464 | 0.015120063 |

Primary displacement uses the supplied BOLD/DWI timing ratio. The alternative
uses the implied ratio for both, scaling displacement by 0.761004606. Registration
is held fixed and FC/nuisance PCs recomputed. This does not refit topup or cover
all uncertainty in its field. Source metadata are preserved unchanged.

BOLD starts 9 min 18 s before DWI pre-op and 10 min 20 s before DWI post-op.
No shim settings or BOLD-specific reversed-PE acquisition are supplied. Rigid
transfer assumes compatible static fields across these scans; drift, shim changes,
motion-dependent susceptibility and GE-EPI dropout remain limitations. Anatomical
overlays and positive Jacobians are QC rather than independent validation.

Tests use a known linear deformation phantom with analytical intensity/Jacobian
and both PE signs, including a wrong-sign control. They check oblique physical
motion composition and finite-difference determinants. Real-data checks verify
finite outputs, unchanged motion/SC/censoring and fresh FC image hashes.

`bold_qc.py` compares all variants on the intersection of valid parcels with fixed
corrected SC. Its component-swap diagnostic crosses old/corrected ROI signals
with old/refitted PCs. The large postoperative mean-FC difference is largely
associated with PC refitting; these noncortical PCs are not validated WM/CSF.
Neither PC set is established as physiological ground truth.

Primary implementation references consulted:

- [FSL topup FAQ: using fields for fMRI](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/FAQ/index.html)
- [NiPreps SDCFlows 2.6.0 transform implementation](https://github.com/nipreps/sdcflows/blob/2.6.0/sdcflows/transform.py)

The project uses its own transparent implementation; fMRIPrep/SDCFlows was not run.

Primary FC now uses retained-frame anatomical WM/CSF five-PC regression;
all BOLD package comparisons use that same recipe. Historical noncortical
PC swaps remain explicitly labelled diagnostics. See PRIMARY_DENOISING.md.
