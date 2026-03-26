#!/usr/bin/env bash
set -euo pipefail

# Download training data for openproof-ml.
# Sources: LeanDojo Benchmark 4, Lean Workbook Plus, Goedel-Pset, MiniF2F

DATA_DIR="${1:-data/raw}"
BENCH_DIR="data/benchmarks"

mkdir -p "$DATA_DIR" "$BENCH_DIR"

echo "=== Downloading training data ==="

# 1. LeanDojo Benchmark 4 (Mathlib4 tactic traces)
if [ ! -d "$DATA_DIR/leandojo" ]; then
    echo "[1/4] Downloading LeanDojo Benchmark 4..."
    # LeanDojo provides data via Zenodo or HuggingFace
    python -c "
from datasets import load_dataset
ds = load_dataset('kaiyuy/leandojo-benchmark-4', split='train')
ds.to_json('$DATA_DIR/leandojo/train.jsonl')
print(f'Downloaded {len(ds)} examples')
"
    echo "  Done."
else
    echo "[1/4] LeanDojo already downloaded, skipping."
fi

# 2. Lean Workbook Plus
if [ ! -d "$DATA_DIR/lean_workbook" ]; then
    echo "[2/4] Downloading Lean Workbook Plus..."
    python -c "
from datasets import load_dataset
ds = load_dataset('internlm/Lean-Workbook', split='train')
ds.to_json('$DATA_DIR/lean_workbook/train.jsonl')
print(f'Downloaded {len(ds)} examples')
"
    echo "  Done."
else
    echo "[2/4] Lean Workbook already downloaded, skipping."
fi

# 3. Goedel-Pset-v1-solved (proofs from Goedel-Prover expert iteration)
if [ ! -d "$DATA_DIR/goedel_pset" ]; then
    echo "[3/4] Downloading Goedel-Pset-v1-solved..."
    python -c "
from datasets import load_dataset
ds = load_dataset('Goedel-LM/Goedel-Pset-v1-solved', split='train')
ds.to_json('$DATA_DIR/goedel_pset/train.jsonl')
print(f'Downloaded {len(ds)} examples')
"
    echo "  Done."
else
    echo "[3/4] Goedel-Pset already downloaded, skipping."
fi

# 4. MiniF2F (for evaluation)
if [ ! -d "$BENCH_DIR/miniF2F-lean4" ]; then
    echo "[4/4] Cloning MiniF2F-lean4..."
    git clone --depth 1 https://github.com/facebookresearch/miniF2F.git "$BENCH_DIR/miniF2F-lean4" 2>/dev/null || \
    git clone --depth 1 https://github.com/rah4927/lean-dojo-mew.git "$BENCH_DIR/miniF2F-lean4" 2>/dev/null || \
    echo "  Warning: MiniF2F clone failed. Copy from openproof/benchmarks/miniF2F-lean4 if available."
    echo "  Done."
else
    echo "[4/4] MiniF2F already present, skipping."
fi

echo ""
echo "=== Download complete ==="
echo "Raw data in: $DATA_DIR"
echo "Benchmarks in: $BENCH_DIR"
echo ""
echo "Next: run 'make extract' to build training JSONL"
