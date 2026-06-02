EDA_DATASET_DIR ?= D:/Dataset/archive
UV ?= uv --cache-dir .uv-cache

setup:
	$(UV) sync
	$(UV) run pre-commit install

run-eda:
	$(UV) run python scripts/run_retailrocket_eda.py --dataset-dir "$(EDA_DATASET_DIR)"

test:
	$(UV) run python -m unittest discover -s tests

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

notebook:
	$(UV) run jupyter notebook notebooks/01_retailrocket_eda.ipynb
