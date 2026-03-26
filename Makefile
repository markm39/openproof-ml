.PHONY: setup setup-lean setup-all download-data extract train-sft train-expert-iter train-dapo eval export test lint format clean

CONFIG ?= configs/sft_qwen35_2b.yaml
LEAN_VERSION ?= v4.28.0
MATHLIB_VERSION ?= v4.28.0

# ── Full pipeline (one command on a fresh instance) ──────────────────

all: setup-all download-data extract train-sft
	@echo "Pipeline complete. Run 'make eval' to benchmark."

# ── Setup ────────────────────────────────────────────────────────────

setup:
	pip install -e ".[dev]"
	pre-commit install || true

setup-lean:
	@echo "=== Installing elan (Lean toolchain manager) ==="
	curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain leanprover/lean4:$(LEAN_VERSION)
	@echo "=== Setting up Lean project with Mathlib ==="
	mkdir -p lean
	@echo 'leanprover/lean4:$(LEAN_VERSION)' > lean/lean-toolchain
	@echo '[package]\nname = "openproof-ml-lean"\nversion = "0.1.0"\n\n[[require]]\nname = "mathlib"\ngit = "https://github.com/leanprover-community/mathlib4.git"\nrev = "$(MATHLIB_VERSION)"\n' > lean/lakefile.toml
	cd lean && ~/.elan/bin/lake update
	@echo "=== Downloading Mathlib cache ==="
	cd lean && ~/.elan/bin/lake exe cache get || true
	@echo "=== Building Pantograph ==="
	mkdir -p vendor
	git clone --depth 1 https://github.com/leanprover/Pantograph.git vendor/Pantograph || true
	echo 'leanprover/lean4:$(LEAN_VERSION)' > vendor/Pantograph/lean-toolchain
	cd vendor/Pantograph && ~/.elan/bin/lake build repl
	@echo "=== Lean + Pantograph setup complete ==="

setup-all: setup setup-lean

# ── Data ─────────────────────────────────────────────────────────────

download-data:
	bash scripts/download_data.sh

extract:
	python scripts/extract_tactics.py \
		--input data/raw \
		--output data/processed/train.jsonl \
		--val-output data/processed/val.jsonl \
		--val-split 0.05

# ── Training ─────────────────────────────────────────────────────────

train-sft:
	python -m openproof_ml.training.sft --config $(CONFIG)

train-expert-iter:
	python -m openproof_ml.training.expert_iteration --config $(CONFIG)

train-dapo:
	python -m openproof_ml.training.dapo --config $(CONFIG)

# ── Evaluation ───────────────────────────────────────────────────────

eval:
	python -m openproof_ml.eval.minif2f --config $(CONFIG)

# ── Export ────────────────────────────────────────────────────────────

export:
	python scripts/export_gguf.py --config $(CONFIG)

# ── Dev ──────────────────────────────────────────────────────────────

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
