# openproof-ml

Training pipeline for on-device Lean 4 tactic prediction models.

Takes a goal state, outputs a tactic. Small enough to run locally (~1-1.5GB quantized), fast enough for real-time proof search (<300ms/tactic on Apple Silicon).

## Overview

This repo trains a step-level tactic model for [OpenProof](https://github.com/markm39/openproof). The model plugs into OpenProof's best-first search via ollama -- zero code changes needed on the inference side.

**Training pipeline:**
1. **SFT** on 1.2M (state, tactic) pairs from Mathlib4, Lean Workbook, and Goedel-Pset
2. **Expert iteration** -- self-play proof search discovers new proofs, generating fresh training data
3. **DAPO RL** -- reinforcement learning with per-tactic Lean compiler feedback

**Base models compared:**
- Qwen3.5-2B (March 2026, hybrid GDN attention -- first application to theorem proving)
- Qwen3-1.7B (proven base, used by Kimina-Prover-RL-1.7B at 76.6% MiniF2F)

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Download training data
make download-data

# Extract (state, tactic) pairs
make extract

# Train (needs GPU -- run on Thunder Compute / cloud)
make train-sft CONFIG=configs/sft_qwen35_2b.yaml

# Evaluate on MiniF2F
make eval CONFIG=configs/eval_minif2f.yaml

# Export to ollama
make export CONFIG=configs/export.yaml
```

## Project structure

```
configs/          YAML configs for each experiment
scripts/          Data download, extraction, export scripts
src/openproof_ml/
  data/           Dataset loading, prompt formatting
  model/          Model wrappers
  training/       SFT, expert iteration, DAPO trainers
  eval/           MiniF2F evaluation harness
  search/         Pantograph client + best-first search (Python)
  utils/          Config loading, logging
tests/            Unit tests
paper/            Paper (LaTeX)
```

## Prompt format

The model uses the BFS-Prover-V2 format:

```
{goal_state}:::
```

Input is the raw Lean goal state (Pantograph `target.pp` format). Output is a single tactic. No chat template.

## Integration with OpenProof

The trained model is served via ollama and consumed by OpenProof's `OllamaProposer`:

```
openproof-ml (training) --> GGUF --> ollama --> openproof (inference)
```

## License

Apache 2.0
