# Interactive notebooks on Binder

[Launch the setup notebook](https://mybinder.org/v2/gh/arvidl/glioma-connectivity-and-cognition/v0.1.2-binder?urlpath=lab/tree/binder/launch.ipynb), wait for JupyterLab, and run all cells in `binder/launch.ipynb`. Select a notebook and click **Download and verify selected runtime**. After **Ready**, open the supplied notebook link and run all its cells. Notebook 04 is a small first example. Return to the launcher before switching to a notebook with different data requirements.

## Environment and downloads

The launch link pins the code to the versioned `v0.1.2-binder` tag. Runtime assets are independently pinned to `v0.1.1-notebooks`. Documentation updates on `main` do not change this environment. Maintainers should create and test a new versioned launch target when updating notebook code or dependencies.

`binder/requirements.txt` installs uv 0.13.1 into repo2docker's Jupyter server environment. `binder/postBuild` runs `uv sync --frozen`: `.python-version` selects Python 3.12.14 and `uv.lock` fixes the project dependencies. The `python3` kernel is registered **inside `.venv/share/jupyter` with its absolute interpreter path**, matching the metadata of all 15 study notebooks. `binder/start` exposes that directory's widget extensions and kernel to the server and limits numerical thread counts. Registering only a user kernel is insufficient: the virtual environment's generic `python3` specification can otherwise shadow it through `JUPYTER_PATH`. The server uses repo2docker's Python 3.12 runtime; the project kernel uses the pinned interpreter.

Opening the launcher and displaying its controls cause no runtime-data downloads. Pressing its download button calls `pat16.notebook_assets.ensure_notebook("04")` (or the selected number). The same operation is available in any notebook cell:

```python
from pat16.notebook_assets import ensure_notebook
ensure_notebook("14")
```

Or in a project terminal:

```sh
uv run python -m pat16.notebook_assets --list
uv run python -m pat16.notebook_assets 04
```

The 15 profiles have different compressed sizes, listed in the launcher. Clinical/graph notebooks are small; imaging and saved resampling arrays can require hundreds of MiB. Only the selected profile is restored. Fully present, verified bundles need no download; partially present bundles are downloaded as a whole while matching existing files are preserved. Compressed packs are discarded after restoration. Interrupted downloads can be retried. A local modification is preserved and raises an explicit error rather than being overwritten.

## Data identity and scope

The supplementary release `v0.1.1-notebooks` contains per-notebook runtime packs. `reproducibility/notebook-assets.json` records every archive's URL, byte count and SHA256, plus the same information for every extracted file. All members are **byte-identical subsets of the original `v0.1.0-assets` release** recorded in `reproducibility/assets.json`. No patient arrays are reduced, substituted or synthesized. The safe extractor accepts only allowlisted regular members under `data/` or `results/`, verifies their hashes and refuses modified destinations, symlinks and unsafe paths. Original source licenses and attribution remain in [THIRD_PARTY.md](../THIRD_PARTY.md).

These packs supply the existing notebook computations and interactive callbacks. They omit unused historical audit trees and expensive reconstruction intermediates; they do not supply the full regression-test or raw MRI reconstruction environment. Use the complete runtime release and [local reproduction guide](REPRODUCING.md) for those routes. FSL and TeX are not installed in Binder. The scientific notebook sources, primary analyses and tracked paper PDF are unchanged by this packaging.

## Session resources

Public Binder is a shared service with [limited memory, session time and bandwidth](https://mybinder.readthedocs.io/en/latest/about/user-guidelines.html). Its documented RAM range is 1–2 GB. Run one study notebook kernel at a time and shut down kernels from JupyterLab's Running panel before switching. Large packs can take a long time over a throttled connection; the download is explicit and its size is shown before requesting it. The recommended small starting notebooks are 03, 04, 05, 08, 09 and 12.

Sessions and downloaded assets are temporary. Download edited notebooks before leaving. If a session ends, launch again and restore the required profile. For larger experiments, repeated tractography, extensive resampling or sustained work, use the local uv environment.

## Building and testing the packs

Maintainers first restore the complete original runtime release. The explicit access lists and additional control-dependent patterns are in `binder/notebook-profiles.json`. Patterns match each path component separately, so archived trees are not accidentally included. Build all packs locally:

```sh
uv run python scripts/build_notebook_assets.py
uv run pytest -q tests/test_notebook_assets.py
uv run python scripts/validate_binder.py --assets-dir dist/notebook-assets
```

The builder checks each source against the original release manifest before packaging it. It writes the new manifest and local archives but never uploads them. Existing archive hashes permit reuse on an unchanged rebuild. The subset-provenance test also runs in lightweight regular CI; it needs no runtime downloads.

`validate_binder.py` creates an isolated directory per notebook, with that notebook's assigned runtime packs and the project sources. Kernels use the locked interpreter and assert that they import the isolated project. It executes the notebook and directly invokes captured widget callbacks with their defaults and each control's choices/endpoints while other controls remain at defaults. Direct invocation exposes errors that widget output handling can hide from headless execution. This is not an exhaustive Cartesian product of all controls. The validator writes `tmp/binder-validation.json`; use `--notebook 04` for one notebook or omit `--assets-dir` to test public downloads.

The Binder configuration follows the official [repo2docker configuration](https://repo2docker.readthedocs.io/en/latest/configuration/) and [postBuild/start hook](https://repo2docker.readthedocs.io/en/latest/configuration/actions/) conventions. A full local repo2docker image build additionally requires a running Docker daemon; Python-only checks do not prove that a public Binder launch has completed.

The checked [validation record](../reproducibility/binder_validation.json) records the tested platform, manifest hash, test totals and isolated notebook/control checks, with their scope and memory-measurement limits.

Validation on 2026-10-18: 82 tests passed; all 15 notebooks (90 original code cells) and 551 direct widget callback calls passed in isolated runtime subsets. All 17 complete public archive downloads matched their sizes and SHA256 hashes. The versioned public Binder image built and launched on the GESIS federation host; Jupyter REST/WebSocket checks verified the pinned kernel, launcher download action, public notebook-04 runtime restoration and notebook execution. Browser frontend rendering was not automated. The temporary test kernel and server were shut down.

## Manual GitHub Actions validation

Open [Binder image validation (manual)](https://github.com/arvidl/glioma-connectivity-and-cognition/actions/workflows/binder-validation.yml) and select **Run workflow**. The workflow has no push, pull-request or scheduled trigger. Regular CI keeps its portable tests and paper compilation; image builds and runtime downloads run only on request.

- **scope:** `all` (default) executes all 15 notebooks with per-control callback checks; `04` runs the small cognition smoke case.
- **binder_ref:** leave blank to build the selected workflow commit. Enter `v0.1.2-binder` to test the published launch environment, or another branch/tag/SHA in this repository. The summary and `source.json` distinguish the image commit from the workflow/test-harness commit.

For example, with the authenticated GitHub CLI:

```sh
gh workflow run binder-validation.yml --ref main -f scope=all
gh workflow run binder-validation.yml --ref main -f scope=04 -f binder_ref=v0.1.2-binder
gh run list --workflow binder-validation.yml --limit 5
gh run download RUN_ID --dir downloaded-binder
```

The workflow uses Ubuntu 24.04 and [repo2docker 2026.4.0](https://repo2docker.readthedocs.io/en/2026.04.0/cli/). A separate pristine checkout supplies the image source, so local runtime caches and the external test harness cannot enter the build context. No image is published. The runner starts the normal image entrypoint and JupyterLab on loopback, with a temporary test token and no GitHub credentials in the container.

`scripts/validate_binder_image.py` checks the server's registered `python3` kernel and the actual interpreter/version through Jupyter REST/WebSocket APIs. It rejects untracked runtime members already present in the image, executes the welcome notebook, and checks that initialization downloads nothing. It invokes the real download callback for notebook 04, checks the exact public URLs requested, verifies archive/member checksums, rejects restoration of unselected runtime members, and confirms that repeating the request causes no download. It then executes notebook 04 in that server kernel.

Next, `scripts/validate_binder.py` runs inside the image's locked environment. Each selected notebook gets a separate temporary project containing only its assigned packs, freshly downloaded without authentication or an asset cache. It executes all original cells and checks widget defaults and each control's choices/endpoints, with other controls at defaults. Its optional `--executed-dir` saves successful study notebooks, excluding test setup/check cells and clearing historical outputs before execution.

Download **`binder-validation-<run-id>-<attempt>`** from the run's **Artifacts** section, or use `gh run download` above. GitHub sign-in is required; retention is seven days. The artifact contains build/server/download logs, exact source/image metadata, `launcher.json`, incremental `notebooks.json` and successfully executed notebooks under `notebooks/`. Failure logs and any earlier successful notebooks are retained, and the container is removed even after test failure. The temporary token is redacted from collected server logs.

This checks an actual image on the Actions runner. It does not test public Binder scheduling or automate browser frontend rendering. The container has a 4 GiB limit to accommodate the validation harness and nested notebook kernels; passing does not establish compliance with public Binder's 1–2 GB limit. Full regression tests and raw reconstruction use the separate full-runtime workflow/local route.
