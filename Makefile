EDA_DATASET_DIR ?= D:/Dataset/archive
PRE_COMMIT_HOME ?= $(CURDIR)/.pre-commit-cache
UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
UV_PYTHON_INSTALL_DIR ?= $(CURDIR)/.uv-python
UV ?= uv --cache-dir '$(UV_CACHE_DIR)'
WITH_LOCAL_CACHE = powershell -NoProfile -Command "$$env:PRE_COMMIT_HOME='$(PRE_COMMIT_HOME)'; $$env:UV_PYTHON_INSTALL_DIR='$(UV_PYTHON_INSTALL_DIR)';

setup:
	$(WITH_LOCAL_CACHE) $(UV) sync"
	git config core.hooksPath .githooks

run-eda:
	$(WITH_LOCAL_CACHE) $(UV) run python scripts/run_retailrocket_eda.py --dataset-dir '$(EDA_DATASET_DIR)'"

test:
	$(WITH_LOCAL_CACHE) $(UV) run python -m unittest discover -s tests"

lint:
	$(WITH_LOCAL_CACHE) $(UV) run ruff check ."

format:
	$(WITH_LOCAL_CACHE) $(UV) run ruff format ."

pre-commit:
	$(WITH_LOCAL_CACHE) $(UV) run pre-commit run --all-files"

notebook:
	$(WITH_LOCAL_CACHE) $(UV) run jupyter notebook notebooks/01_retailrocket_eda.ipynb"
