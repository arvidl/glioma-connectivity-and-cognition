# Reproducing the PAT16 study

The public repository is https://github.com/arvidl/glioma-connectivity-and-cognition. Runtime data and analysis checkpoints are available from release [v0.1.0-assets](https://github.com/arvidl/glioma-connectivity-and-cognition/releases/tag/v0.1.0-assets). `reproducibility/assets.json` records exact public URLs, sizes and SHA256 checksums. The fetcher verifies the real files and never substitutes synthetic data. The public clone/download checks are recorded in `reproducibility/public_validation.json`.

## 1. Source-only clone: portable tests and paper

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a TeX distribution with latexmk, pdfLaTeX, BibTeX, lmodern and standard LaTeX packages (MacTeX on macOS; Ubuntu packages `latexmk texlive-latex-extra texlive-fonts-recommended lmodern`). TeX and FSL are not Python dependencies. Python 3.12.14 is pinned; `uv.lock` fixes the tested dependency resolution.

Clone the public repository:

```sh
git clone https://github.com/arvidl/glioma-connectivity-and-cognition.git
cd glioma-connectivity-and-cognition
uv sync --frozen
make unit
make paper
git ls-files --error-unmatch report/main.pdf
```

The paper PDF is an ordinary tracked Git blob, not an LFS pointer. All report figures, generated tables, TeX and bibliography are tracked, so `make paper` needs neither MRI assets nor Python. Do not delete `report/generated` or `report/figures` for this route. PDF bytes can vary with TeX metadata/version; compare scientific content rather than assuming universal binary identity.

Regular push/pull-request CI runs portable scientific/extraction contracts and source-only TeX compilation, without runtime downloads. The separate [Full runtime validation (manual) workflow](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/runtime-validation.yml) is opt-in: choose **Run workflow** in GitHub Actions, or use the GitHub CLI:

```sh
gh workflow run runtime-validation.yml --ref main
```

Each manual run installs the frozen uv environment, downloads all seven public release parts (about 4.9 GiB, without an asset cache or download token), verifies part and extracted-member SHA256 checksums plus the raw-input manifest, runs **all pytest tests**, and executes **all 15 notebooks** from the saved runtime checkpoints. It uses an Ubuntu runner with a 60-minute limit; only Python dependencies are cached. The run summary records the commit, release, step outcomes and test/notebook counts. Logs, JUnit XML and successfully executed notebooks are downloadable as a workflow artifact for seven days (GitHub sign-in is required for Actions artifact downloads); failures retain the available diagnostic logs. Executed notebooks are uploaded only after the whole notebook batch succeeds, and the historical execution record is removed before starting that batch. Neither workflow reruns raw FSL preprocessing; paper compilation remains in regular CI. The scientific release snapshot, tracked paper and local validation records are unchanged by a manual run.

## 2. Restore exact runtime assets

Git tracks source, documentation, executed notebooks, small current summaries, paper and its generated figures/tables. NIfTI images, matrices, saved temporal/tractography draws, intermediate corrections and immutable historical snapshots are outside Git. `scripts/build_release_assets.py` bundles `data/` and `results/`, hashes each member and each compressed part, and splits at 900 MiB. It never uploads. The seven public release parts total about 4.9 GiB. Each bundle is one gzip/tar stream split into ordered parts; individual parts are not separate archives. The fetcher handles concatenation and extraction.

Download directly from the public release (no GitHub account or token is required):

```sh
python3 scripts/fetch_data.py
python3 scripts/fetch_data.py --verify-only
uv run python scripts/manifest.py --verify
```

`--profile inputs` restores only raw/reference inputs; the default `all` additionally restores analysis/historical checkpoints. The fetcher checks all part hashes before extraction, accepts only allowlisted regular paths under `data/` or `results/`, verifies every extracted member, refuses symlinks/traversal/duplicates and preserves modified destination files. Use `--overwrite` only when deliberately replacing locally changed runtime data. Repeating restoration of identical files is safe.

Downloads are cached in `.cache/assets`; verified completed parts are reused on rerun. An interrupted part is downloaded again. For offline restoration, copy all seven parts to a local directory and use `python3 scripts/fetch_data.py --assets-dir /absolute/path/to/assets`. No sibling repository is consulted. Raw data license/provenance remains in `data/manifest.json`, BIDS sidecars and `THIRD_PARTY.md`.

## 3. Replay analyses and all notebooks

```sh
uv sync --frozen
make test
uv run python scripts/run_all.py --notebooks --report
uv run python scripts/record_analysis.py
uv run jupyter lab notebooks/
```

This route reuses verified imaging and tractography checkpoints. It recomputes downstream summaries/figures and verifies or reuses the expensive sampling draws. Fifteen notebooks use this clone's installed package and project-local kernel; live widgets initialize on Run All. Python dependencies are installed once; the saved scientific workflows then run offline. Jupyter requires local loopback sockets.

Primary-only reproduction and figure regeneration:

```sh
uv run python scripts/recommend_primary_fc.py --force
uv run python scripts/plot_primary_fc.py
uv run python scripts/build_notebooks.py
uv run python scripts/execute_notebooks.py
uv run python scripts/report_tables.py
make paper
```

The first command recomputes all nine common-duration fraction/block designs, each paired across frozen/coefficient/full refits. It preserves preceding analyses. Core design/output hashes are in `results/primary_fc/design.json`; `main_summary.json` is the current primary resampling entry. The original observed `results/summary.json` and network points remain unchanged. Complete clinical values/dictionaries and historical evidence are preserved rather than overwritten.

`results/analysis_provenance.json` fingerprints delivered scientific sources and outputs. Historical checkpoints bind earlier source versions, and archived manifests explicitly distinguish immutable inputs from archived former mutable files. Recomputing notebooks updates execution timings and output metadata, so regenerate current provenance after an intentional rerun. Historical validators document prior deliveries and must not overwrite their archived audit records. `scripts/validate_primary_fc.py` audits the delivered snapshot, including its PDF hash and bundled render/test/pipeline records in `results/primary_fc_audit/`. An intentional new TeX build can change the PDF bytes: review/render that PDF and refresh its audit records before claiming a new delivery audit. The numerical tests and notebook executor do not require a prior render record.

## 4. Reconstruct from raw MRI

Install FSL separately with `topup`, `eddy_cpu` and `fast` available, then set `FSLDIR` to that installation and include its `share/fsl/bin` in PATH. The studied machine uses `/Users/arvid/fsl`, but code discovers FSL via `FSLDIR`/PATH rather than requiring this username. An external FSL installation/license is required; binaries are not bundled. See [FSL licensing](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html).

```sh
export FSLDIR=/absolute/path/to/fsl
export PATH="$FSLDIR/share/fsl/bin:$PATH"
uv run python scripts/run_all.py --from-raw --force --notebooks --report
```

This invalidates base anatomy, correction, motion and connectome completion markers and refits tissue segmentation. Derived stages compare input fingerprints and reuse unchanged checkpoints. It is not an unconditional deletion of every historical or stochastic artifact. To reconstruct all repeat-tractography or sampling checkpoints deliberately, use their individual scripts' `--force` options (inspect `--help`); preserve the supplied historical archives for the regression tests. No raw workflow should silently replace clinical source fields.

A 128-GB Apple Silicon laptop is suitable; no GPU is required. Keep at least 25 GB free for the clone, restored runtime data, dependencies, asset parts and regeneration; larger temporary raw-processing outputs need additional space. Runtime varies with FSL build, thread count, cache state and platform. Exact seeds, parameters and original-time indices are saved. Floating-point/registration/tracking results may differ across software platforms, so new raw runs require QC and scientific tolerance checks, not an assumption of universal binary identity. Susceptibility field transfer, mask quality, tractography specificity and physiological validity remain limitations even when numerical tests pass.

## 5. Release preparation and validation

```sh
uv run python scripts/validate_primary_fc.py
uv run python scripts/record_analysis.py
uv run python scripts/build_release_assets.py
```

Build assets **after** final scientific edits/execution/provenance. The asset manifest lives outside that scientific provenance to avoid circular self-hashes. The clone-validation record is `reproducibility/local_validation.json`; it records fresh-environment setup, source-only checks/compilation, checksummed asset restoration, complete tests and notebook execution. Logs and render contacts remain in ignored `tmp/` during development. Git-tracked results are compact public summaries; full runtime and audit history are restored from the assets.

The owner authorized public publication on 2026-10-18. The `published` flag is enabled only after the exact release parts are uploaded and available. `reproducibility/public_validation.json` records the separately checked anonymous clone, public asset downloads, checksum verification, tests, notebooks and paper build; `local_validation.json` describes the preceding local preparation and is retained as history. Release verification metadata lives outside the scientific provenance to avoid circular hashes. Licenses/attribution are separated in `THIRD_PARTY.md`; citation metadata is in `CITATION.cff`. No new data consent/ethics approval is asserted by packaging the already public source material.
