.PHONY: setup download-data extract train-sft train-dapo eval export test lint clean

CONFIG ?= configs/sft_qwen35_2b.yaml

setup:
	pip install -e ".[dev]"
	pre-commit install

download-data:
	bash scripts/download_data.sh

extract:
	python scripts/extract_tactics.py \
		--input data/raw \
		--output data/processed/train.jsonl \
		--val-output data/processed/val.jsonl \
		--val-split 0.05

train-sft:
	python -m openproof_ml.training.sft --config $(CONFIG)

train-expert-iter:
	python -m openproof_ml.training.expert_iteration --config $(CONFIG)

train-dapo:
	python -m openproof_ml.training.dapo --config $(CONFIG)

eval:
	python -m openproof_ml.eval.minif2f --config $(CONFIG)

export:
	python scripts/export_gguf.py --config $(CONFIG)

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ scripts/ tests/
	ruff format --check src/ scripts/ tests/

format:
	ruff check --fix src/ scripts/ tests/
	ruff format src/ scripts/ tests/

clean:
	rm -rf checkpoints/ outputs/ wandb/ __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
