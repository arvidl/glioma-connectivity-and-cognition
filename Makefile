.PHONY: install unit data verify test analysis notebooks paper
install:
	uv sync --frozen
unit:
	uv run pytest -q tests/test_scientific_contracts.py tests/test_public_repository.py tests/test_pc_sampling.py::test_nonoverlapping_tiles_exact_censoring_targets_and_replay
data:
	python3 scripts/fetch_data.py $(if $(ASSETS_DIR),--assets-dir "$(ASSETS_DIR)",)
verify:
	python3 scripts/fetch_data.py --verify-only
	uv run python scripts/manifest.py --verify
test:
	uv run pytest -q
analysis:
	uv run python scripts/run_all.py
notebooks:
	uv run python scripts/execute_notebooks.py
paper:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
