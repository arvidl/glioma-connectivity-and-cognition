# Glioma connectivity and cognition: PAT16

A didactic, exploratory paired case study of glioma, structural/functional connectivity, CANTAB and psychosocial context. Python with **uv**, 15 interactive Jupyter notebooks, reproducible analyses and a LaTeX/BibTeX report with **`report/main.pdf` tracked by ordinary Git**. No sibling project, cloud service or GPU is needed after setup and runtime-asset restoration.

**Public release:** [source repository](https://github.com/arvidl/glioma-connectivity-and-cognition), [tracked paper PDF](report/main.pdf) and [runtime assets v0.1.0-assets](https://github.com/arvidl/glioma-connectivity-and-cognition/releases/tag/v0.1.0-assets). Large inputs/checkpoints are seven checksummed release parts (about 4.9 GiB); `reproducibility/assets.json` records every part and restored file. A source clone can compile the paper and run portable tests immediately; the full experiments additionally require those assets. Publication verification is recorded in `reproducibility/public_validation.json`; the preceding local validation remains a dated preparation record.

## Recommended primary analyses

- **Denoising:** five PCs from session-specific anatomical WM/CSF cores; temporal projection, voxel scaling and PCA trained on retained original frames, followed by joint 24-motion/trend/DCT regression. No GSR or low-pass. [Rationale](config/PRIMARY_DENOISING.md).
- **SC:** each session's 16/20 tractography consensus, all-run mean counts including zero per 10,000 attempted seeds. Graphs/network means use own supports; paired SC–FC coupling uses the fixed 823-edge intersection and session-specific weights. [Rationale](config/PRIMARY_SC.md).
- **FC stability:** nonreplacement original-time block subsets with **97 distinct frames per session**, ten-frame tiles (24 s), complete WM/CSF nuisance training/regression refitted in each of 200 draws. Cross with the same 500 whole-run SC draws. [Policy fixed before these new contrasts](config/PRIMARY_FC_RESAMPLING.md).

Nonreplacement removes repeated-row weighting of PCA; full refitting includes training sensitivity; a common count matches retained duration. Two-thirds of the smaller eligible pool gives 97 frames, 3.88 minutes and 58 algebraic residual directions after rank-39 regression. This is a **post hoc engineering recommendation**, not an optimality result or independent validation. The target is conditional stability of shorter-sample estimators: **the ranges are not full-run confidence intervals**. Motion is not matched in the primary subsets; previous duration/FD/run-half controls remain sensitivities.

Observed full-run coupling remains **0.18822 → 0.14649 (Δ −0.04173)**. Primary subset median Δ is **−0.04694**, with raw 2.5/97.5 percentiles **[−0.10261, +0.00832]**. FC accounts for 98.33% of the implemented conditional crossed-grid variance. Overlapping subsets/grid cells are dependent; their positive fraction is not a biological probability or p-value. See `results/primary_fc/main_summary.json`.

All alternatives remain sensitivities: 73/97/116 common frames, 5/10/20-frame blocks, frozen/coefficient-only fits, support reselection, repeated-row bootstrap, exact unit coverage, independently matched bootstrap-distinct counts, original session-specific 80% retention, strict censoring and 68-frame duration/FD/run-half matching. Historical evidence is preserved in immutable analysis assets and [previous analysis notes](docs/PREVIOUS_ANALYSES.md).

## Run locally

```sh
git clone https://github.com/arvidl/glioma-connectivity-and-cognition.git
cd glioma-connectivity-and-cognition
uv sync --frozen
make unit
make paper
# Download, checksum and restore the public runtime assets:
python3 scripts/fetch_data.py
make verify
make test
uv run python scripts/run_all.py --notebooks --report
uv run jupyter lab notebooks/
```

A fresh clone must restore the runtime assets before running all tests/notebooks. [Complete reproduction instructions](docs/REPRODUCING.md) distinguish portable checks, saved-analysis replay and raw reconstruction; they cover FSL, TeX, disk use, checksums and public-clone verification. TeX is external to uv. FSL topup/eddy_cpu/FAST are required for raw reconstruction or new tissue segmentation, and are not redistributed.

Regular CI runs portable tests and compiles the paper without downloading runtime assets. Use [Full runtime validation (manual)](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/runtime-validation.yml) → **Run workflow** to download/verify the public assets, run all tests and execute all 15 notebooks on Ubuntu. Logs, test results and executed notebooks are retained as a downloadable Actions artifact for seven days; see the [workflow instructions](docs/REPRODUCING.md#1-source-only-clone-portable-tests-and-paper).

## Clinical interpretation

All **28 CANTAB and 19 psychosocial/context fields** retain original values, units, missingness and coding caveats. MOT becomes slower, RTI faster and SSP span 6→7; SOC/RVP are mixed. Anxiety falls, worry/loneliness rise and employment changes to sick leave. There is no supplied dedicated QoL instrument, retest norm, reliable-change estimate or dated treatment/assessment timeline.

Primary sensorimotor FC range [−0.2064,−0.0170] is negative, but a previous matched-count design crossed zero. Visual [−0.1594,+0.1081], control [−0.1312,+0.1283] and default [−0.1649,+0.0058] cross zero. The visual subset median −0.0209 is much closer to observed −0.0045 than the repeated-row median −0.1993. Network signs depend on sampling and training; seven exploratory network summaries do not diagnose a task deficit, compensation, a surgery effect or a QoL trajectory. No two-observation clinical correlation or composite is fitted. Report Appendices A1/A2 preserve the complete measure dictionaries.

## Notebook guide

| Notebook | Topic |
|---|---|
| 00 | Data, anatomy, registration and evidence boundaries |
| 01 | AP/PA topup/eddy, diffusion models and tractography |
| 02 | BOLD correction, motion, anatomical nuisance regression and FC |
| 03 | Longitudinal graphs, networks and SC–FC coupling |
| 04 | Complete CANTAB and lived-experience fields |
| 05 | Hypothetical growth, injury and graph dynamics |
| 06 | PC refitting mechanism and anatomical denoising comparisons |
| 07 | Censoring, duration/FD/run-half matching |
| 08 | Edge/support/network accounting of historical reversal |
| 09 | Repeated tractography and edge reliability |
| 10 | Joint SC/FC bootstrap and matched sampling sensitivities |
| 11 | Frozen/coefficient/full nuisance bootstrap refits |
| 12 | Clinical/context ledger with current primary network ranges |
| 13 | PC training, multiplicity and nonreplacement controls |
| **14** | **Recommended primary FC stability, sensitivities and clinical integration** |

Live controls initialize with Run All in Jupyter; saved outputs contain offline Plotly JavaScript. No synthetic fallback replaces missing patient data. Simulated growth/injury/dynamics are labelled hypothetical.

## Reuse and citation

Code: MIT. Original explanatory text/figures: CC BY 4.0. Patient data and third-party atlas/templates have separate terms and attribution: [THIRD_PARTY.md](THIRD_PARTY.md), [LICENSE](LICENSE), [LICENSE-CONTENT.md](LICENSE-CONTENT.md), and the input notices. Cite the source BTC datasets and methods as well as this project; see [CITATION.cff](CITATION.cff) and `report/references.bib`.

This is a research/teaching case study with substantial imaging, physiological and clinical uncertainty. It makes no new ethics/consent, predictive or clinical-validation claim.
