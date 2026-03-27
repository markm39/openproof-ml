#!/usr/bin/env bash
set -euo pipefail

# Download training data for openproof-ml.
# Fully self-contained -- run on a fresh GPU instance.
#
# Sources:
#   - LeanDojo (Mathlib4 tactic traces): ~259K tactics from 122K theorems
#   - Lean Workbook (competition math): ~83K problems
#   - Goedel-Pset-v1-solved (large-scale proofs): 1.73M statements, proofs for ~800K
#   - MiniF2F-lean4 (evaluation benchmark): 244 test problems

DATA_DIR="${1:-data/raw}"
BENCH_DIR="data/benchmarks"

mkdir -p "$DATA_DIR"/{leandojo,lean_workbook,goedel_pset} "$BENCH_DIR"

echo "=== Downloading training data ==="
echo "This may take 15-30 minutes depending on network speed."
echo ""

# 1. LeanDojo tactic data (via tasksource mirror on HuggingFace)
if [ ! -f "$DATA_DIR/leandojo/train.jsonl" ]; then
    echo "[1/4] Downloading LeanDojo tactic traces..."
    python3 -c "
import os, json
os.makedirs('$DATA_DIR/leandojo', exist_ok=True)
from datasets import load_dataset
# tasksource/leandojo is a HuggingFace mirror of the LeanDojo benchmark
try:
    ds = load_dataset('tasksource/leandojo', split='train')
    ds.to_json('$DATA_DIR/leandojo/train.jsonl')
    print(f'  Downloaded {len(ds)} examples from tasksource/leandojo')
except Exception as e:
    print(f'  tasksource/leandojo failed ({e}), trying LeanDojo direct download...')
    # Fallback: use LeanDojo's own download script
    import subprocess
    subprocess.run([
        'python3', '-c',
        '''
import os
os.makedirs(\"$DATA_DIR/leandojo\", exist_ok=True)
# LeanDojo distributes via their Python package
try:
    from lean_dojo import LeanGitRepo, trace
    print(\"Using LeanDojo package for data extraction\")
except ImportError:
    print(\"Install lean-dojo: pip install lean-dojo\")
    print(\"Or download manually from https://zenodo.org/records/12818690\")
'''
    ], check=False)
"
    echo "  Done."
else
    echo "[1/4] LeanDojo already downloaded, skipping."
fi

# 2. Lean Workbook (competition-level formalized problems)
if [ ! -f "$DATA_DIR/lean_workbook/train.jsonl" ]; then
    echo "[2/4] Downloading Lean Workbook..."
    python3 -c "
import os
os.makedirs('$DATA_DIR/lean_workbook', exist_ok=True)
from datasets import load_dataset
ds = load_dataset('internlm/Lean-Workbook', split='train')
ds.to_json('$DATA_DIR/lean_workbook/train.jsonl')
print(f'  Downloaded {len(ds)} examples')
"
    echo "  Done."
else
    echo "[2/4] Lean Workbook already downloaded, skipping."
fi

# 3. Goedel-Pset-v1-solved (large-scale proofs from expert iteration)
if [ ! -f "$DATA_DIR/goedel_pset/train.jsonl" ]; then
    echo "[3/4] Downloading Goedel-LM proof data..."
    python3 -c "
import os
os.makedirs('$DATA_DIR/goedel_pset', exist_ok=True)
from datasets import load_dataset

# Lean-workbook-proofs: 29.7K solved problems with full proofs
print('  Downloading Goedel-LM/Lean-workbook-proofs...')
ds = load_dataset('Goedel-LM/Lean-workbook-proofs', split='train')
ds.to_json('$DATA_DIR/goedel_pset/train.jsonl')
print(f'  Downloaded {len(ds)} workbook proofs')

# Goedel-Pset-v1: 1.73M formalized statements (for expert iteration later)
print('  Downloading Goedel-LM/Goedel-Pset-v1 (statements)...')
try:
    ds2 = load_dataset('Goedel-LM/Goedel-Pset-v1', split='train')
    ds2.to_json('$DATA_DIR/goedel_pset/statements.jsonl')
    print(f'  Downloaded {len(ds2)} statements')
except Exception as e:
    print(f'  Goedel-Pset-v1 statements failed: {e} (optional, needed for expert iteration)')
"
    echo "  Done."
else
    echo "[3/4] Goedel-Pset already downloaded, skipping."
fi

# 4. MiniF2F (evaluation benchmark)
if [ ! -d "$BENCH_DIR/miniF2F-lean4" ]; then
    echo "[4/4] Downloading MiniF2F-lean4..."
    # Try the kimina-lean-server repo which has a maintained Lean 4 version
    git clone --depth 1 https://github.com/yangky11/miniF2F-lean4.git "$BENCH_DIR/miniF2F-lean4" 2>/dev/null || \
    echo "  Warning: MiniF2F clone failed. Evaluation will not work."
    echo "  Done."
else
    echo "[4/4] MiniF2F already present, skipping."
fi

echo ""
echo "=== Download complete ==="
du -sh "$DATA_DIR"/* "$BENCH_DIR"/* 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  make extract     # build training JSONL"
echo "  make train-sft   # start training"
