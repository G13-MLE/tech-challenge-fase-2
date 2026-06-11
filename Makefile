.PHONY: run run-cf install lint test download

install:
	uv sync

download:
	kaggle datasets download -d retailrocket/ecommerce-dataset -p tmp/ --unzip

run:
	uv run python prototype.py

run-cf:
	uv run python prototype_cf.py

lint:
	uv run ruff check .

test:
	@echo "Nenhum teste configurado ainda"