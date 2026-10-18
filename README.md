# Glioma connectivity and cognition: PAT16

[![Regular CI](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/ci.yml)
[![Full runtime validation (manual)](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/runtime-validation.yml/badge.svg?branch=main&event=workflow_dispatch)](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/runtime-validation.yml)
[![Binder image validation (manual)](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/binder-validation.yml/badge.svg?branch=main&event=workflow_dispatch)](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/binder-validation.yml)
[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arvidl/glioma-connectivity-and-cognition/v0.1.2-binder?urlpath=lab/tree/binder/launch.ipynb)

A didactic, exploratory paired case study of glioma, structural/functional connectivity, CANTAB and psychosocial context. Python with **uv**, 15 interactive Jupyter notebooks, reproducible analyses and a LaTeX/BibTeX report with **`report/main.pdf` tracked by ordinary Git**. No sibling project, cloud service or GPU is needed after setup and runtime-asset restoration.

**Public release:** [source repository](https://github.com/arvidl/glioma-connectivity-and-cognition), [tracked paper PDF](report/main.pdf) and [runtime assets v0.1.0-assets](https://github.com/arvidl/glioma-connectivity-and-cognition/releases/tag/v0.1.0-assets). Large inputs/checkpoints are seven checksummed release parts (about 4.9 GiB); `reproducibility/assets.json` records every part and restored file. A source clone can compile the paper and run portable tests immediately; the full experiments additionally require those assets. Publication verification is recorded in `reproducibility/public_validation.json`; the preceding local validation remains a dated preparation record.

## Recommended primary analyses

- **Denoising:** five PCs from session-specific anatomical WM/CSF cores; temporal projection, voxel scaling and PCA trained on retained original frames, followed by joint 24-motion/trend/DCT regression. No GSR or low-pass. [Rationale](config/PRIMARY_DENOISING.md).
- **SC:** each session's 16/20 tractography consensus, all-run mean counts including zero per 10,000 attempted seeds. Graphs/network means use own supports; paired SC–FC coupling uses the fixed 823-edge intersection and session-specific weights. [Rationale](config/PRIMARY_SC.md).
- **FC stability:** nonreplacement original-time block subsets with **97 distinct frames per session**, ten-frame tiles (24 s), complete WM/CSF nuisance training/regression refitted in each of 200 draws. Cross with the same 500 whole-run SC draws. [Policy fixed before these new contrasts](config/PRIMARY_FC_RESAMPLING.md).

Nonreplacement removes repeated-row weighting of PCA; full refitting includes training sensitivity; a common count matches retained duration. Two-thirds of the smaller eligible pool gives 97 frames, 3.88 minutes and 58 algebraic residual directions after rank-39 regression. This is a **post hoc engineering recommendation**, not an optimality result or independent validation. The target is conditional stability of shorter-sample estimators: **the ranges are not full-run confidence intervals**. Motion is not matched in the primary subsets; previous duration/FD/run-half controls remain sensitivities.

Observed full-run coupling remains **0.18822 → 0.14649 (Δ −0.04173)**. Primary subset median Δ is **−0.04694**, with raw 2.5/97.5 percentiles **[−0.10261, +0.00832]**. FC accounts for 98.33% of the implemented conditional crossed-grid variance. Overlapping subsets/grid cells are dependent; their positive fraction is not a biological probability or p-value. See `results/primary_fc/main_summary.json`.

All alternatives remain sensitivities: 73/97/116 common frames, 5/10/20-frame blocks, frozen/coefficient-only fits, support reselection, repeated-row bootstrap, exact unit coverage, independently matched bootstrap-distinct counts, original session-specific 80% retention, strict censoring and 68-frame duration/FD/run-half matching. Historical evidence is preserved in immutable analysis assets and [previous analysis notes](docs/PREVIOUS_ANALYSES.md).

## Run interactively on Binder

1. Click **Launch Binder** above. The first build can take several minutes; it opens the [setup notebook](binder/launch.ipynb) in JupyterLab with the locked PAT16 Python kernel.
2. Choose **Run → Run All Cells** in the setup notebook, select a study notebook, then click **Download and verify selected runtime**. Start with **04 (cognition)**, a download of about 0.17 MiB. Runtime release assets are fetched only when you request them; image building and session startup do not download these packs.
3. Once **Ready** appears, follow the notebook link and choose **Run All Cells** there to initialize its interactive controls. Return to the setup notebook to request another notebook's assets.

The per-notebook packs contain exact subsets of the original real-data release, with archive and extracted-file SHA256 checks. The launcher shows download sizes; imaging and resampling notebooks need much larger packs and may download slowly. Binder has [limited memory and temporary sessions](https://mybinder.readthedocs.io/en/latest/about/user-guidelines.html): shut down previous notebook kernels and download edits you want to keep. Local reproduction below supplies full regression tests, raw FSL processing and paper compilation. See [Binder setup, asset provenance and testing](docs/BINDER.md).

Maintainers can run [Binder image validation (manual)](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/binder-validation.yml) to build the image, test the live Jupyter launcher and public downloads, and execute all 15 notebooks from isolated runtime packs. A notebook-04 smoke option and an optional image-code ref are available. Results, logs and executed notebooks are retained for seven days; see [download instructions](#binder-image-validation) and the [workflow instructions](docs/BINDER.md#manual-github-actions-validation). This workflow runs only when dispatched.

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

The status badges link to their workflows and show results on `main`. Both manual-validation badges reflect the latest dispatched run and can cover an earlier commit; neither reruns automatically after a push. For Binder, check the run summary for `scope=all` versus the notebook-04 smoke test, and for the exact image and test-harness commits: an optional `binder_ref` can select a different image source, including the versioned launch environment.

### Download executed notebooks

#### Full runtime validation

1. Open [Full runtime validation (manual)](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/runtime-validation.yml) and select a successful run on `main`. Maintainers can choose **Run workflow** to create a fresh one.
2. In the run summary, scroll to **Artifacts** and download **`runtime-validation-<run-id>-<attempt>`**. GitHub sign-in is required, and artifacts expire after seven days.
3. Extract the ZIP and open its **`notebooks/`** folder for all 15 executed `.ipynb` files. Logs, `pytest.xml` and `notebook_execution.json` are included alongside them. Saved outputs can be viewed directly; rerunning cells or using live controls requires the local environment and runtime assets described above, or the Binder setup.

With the authenticated [GitHub CLI](https://cli.github.com/), list successful runs, replace `RUN_ID` with the chosen ID, and download its artifacts:

```sh
gh run list --repo arvidl/glioma-connectivity-and-cognition --workflow runtime-validation.yml --branch main --status success --limit 5
gh run download RUN_ID --repo arvidl/glioma-connectivity-and-cognition --dir downloaded-runtime
```

The CLI places each artifact in its own directory under `downloaded-runtime/`; the executed notebooks are inside that artifact's `notebooks/` folder. Failed runs retain available diagnostic logs, while executed notebooks are uploaded only after the complete notebook batch succeeds.

#### Binder image validation

1. Open [Binder image validation (manual)](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/binder-validation.yml) and select a successful run on `main`. Check its summary: `scope=all` covers all 15 notebooks; `scope=04` covers only the cognition notebook. Maintainers can choose **Run workflow** with `scope=all` for a fresh complete run.
2. In the run summary, scroll to **Artifacts** and download **`binder-validation-<run-id>-<attempt>`**. GitHub sign-in is required, and artifacts expire after seven days.
3. Extract the ZIP and open **`notebooks/`** for the executed `.ipynb` files. **`source.json`** records the exact image and test-harness commits and selected scope; **`launcher.json`**, **`notebooks.json`** and the logs record validation results. Saved outputs are viewable directly; live controls require running the cells in Binder or the local environment.

With the authenticated GitHub CLI, replace `RUN_ID` with the chosen successful run:

```sh
gh run list --repo arvidl/glioma-connectivity-and-cognition --workflow binder-validation.yml --branch main --event workflow_dispatch --status success --limit 5
gh run download RUN_ID --repo arvidl/glioma-connectivity-and-cognition --dir downloaded-binder
```

The CLI places the artifact under `downloaded-binder/`, with executed notebooks in its `notebooks/` folder. Failed Binder runs retain diagnostic logs and any notebooks that completed successfully before the failure. See the [detailed validation procedure](docs/BINDER.md#manual-github-actions-validation).

## Clinical interpretation

All **28 CANTAB and 19 psychosocial/context fields** retain original values, units, missingness and coding caveats. MOT becomes slower, RTI faster and SSP span 6→7; SOC/RVP are mixed. Anxiety falls, worry/loneliness rise and employment changes to sick leave. There is no supplied dedicated QoL instrument, retest norm, reliable-change estimate or dated treatment/assessment timeline.

Primary sensorimotor FC range [−0.2064,−0.0170] is negative, but a previous matched-count design crossed zero. Visual [−0.1594,+0.1081], control [−0.1312,+0.1283] and default [−0.1649,+0.0058] cross zero. The visual subset median −0.0209 is much closer to observed −0.0045 than the repeated-row median −0.1993. Network signs depend on sampling and training; seven exploratory network summaries do not diagnose a task deficit, compensation, a surgery effect or a QoL trajectory. No two-observation clinical correlation or composite is fitted. Report Appendices A1/A2 preserve the complete measure dictionaries.

## Notebook guide

Each notebook number opens the file on GitHub; **nbviewer** renders the committed notebook and its saved outputs from `main` without running code. For outputs from a particular manual validation run, see [Download executed notebooks](#download-executed-notebooks).

| Notebook (GitHub) | Topic | Executed outputs |
|---|---|---|
| [00](notebooks/00_data_anatomy_and_design.ipynb) | Data, anatomy, registration and evidence boundaries | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/00_data_anatomy_and_design.ipynb) |
| [01](notebooks/01_diffusion_structural_connectivity.ipynb) | AP/PA topup/eddy, diffusion models and tractography | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/01_diffusion_structural_connectivity.ipynb) |
| [02](notebooks/02_bold_functional_connectivity.ipynb) | BOLD correction, motion, anatomical nuisance regression and FC | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/02_bold_functional_connectivity.ipynb) |
| [03](notebooks/03_longitudinal_graphs_and_networks.ipynb) | Longitudinal graphs, networks and SC–FC coupling | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/03_longitudinal_graphs_and_networks.ipynb) |
| [04](notebooks/04_cognition_and_lived_experience.ipynb) | Complete CANTAB and lived-experience fields | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/04_cognition_and_lived_experience.ipynb) |
| [05](notebooks/05_simulations_growth_surgery_and_dynamics.ipynb) | Hypothetical growth, injury and graph dynamics | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/05_simulations_growth_surgery_and_dynamics.ipynb) |
| [06](notebooks/06_nuisance_refitting_and_anatomical_denoising.ipynb) | PC refitting mechanism and anatomical denoising comparisons | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/06_nuisance_refitting_and_anatomical_denoising.ipynb) |
| [07](notebooks/07_censoring_motion_duration_and_coupling.ipynb) | Censoring, duration/FD/run-half matching | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/07_censoring_motion_duration_and_coupling.ipynb) |
| [08](notebooks/08_edges_networks_and_coupling_reversal.ipynb) | Edge/support/network accounting of historical reversal | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/08_edges_networks_and_coupling_reversal.ipynb) |
| [09](notebooks/09_structural_edge_reliability.ipynb) | Repeated tractography and edge reliability | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/09_structural_edge_reliability.ipynb) |
| [10](notebooks/10_joint_sampling_uncertainty.ipynb) | Joint SC/FC bootstrap and matched sampling sensitivities | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/10_joint_sampling_uncertainty.ipynb) |
| [11](notebooks/11_full_run_nuisance_refit.ipynb) | Frozen/coefficient/full nuisance bootstrap refits | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/11_full_run_nuisance_refit.ipynb) |
| [12](notebooks/12_connectivity_cognition_context.ipynb) | Clinical/context ledger with current primary network ranges | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/12_connectivity_cognition_context.ipynb) |
| [13](notebooks/13_pc_sampling_mechanisms.ipynb) | PC training, multiplicity and nonreplacement controls | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/13_pc_sampling_mechanisms.ipynb) |
| **[14](notebooks/14_primary_fc_resampling.ipynb)** | **Recommended primary FC stability, sensitivities and clinical integration** | [nbviewer](https://nbviewer.org/github/arvidl/glioma-connectivity-and-cognition/blob/main/notebooks/14_primary_fc_resampling.ipynb) |

Live controls initialize with Run All in local Jupyter or Binder; nbviewer does not provide a Python kernel. Saved outputs contain offline Plotly JavaScript. No synthetic fallback replaces missing patient data. Simulated growth/injury/dynamics are labelled hypothetical.

## Reuse and citation

Code: MIT. Original explanatory text/figures: CC BY 4.0. Patient data and third-party atlas/templates have separate terms and attribution: [THIRD_PARTY.md](THIRD_PARTY.md), [LICENSE](LICENSE), [LICENSE-CONTENT.md](LICENSE-CONTENT.md), and the input notices. Cite the source BTC datasets and methods as well as this project; see [CITATION.cff](CITATION.cff) and `report/references.bib`.

This is a research/teaching case study with substantial imaging, physiological and clinical uncertainty. It makes no new ethics/consent, predictive or clinical-validation claim.
