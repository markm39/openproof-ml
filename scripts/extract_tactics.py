#!/usr/bin/env python3
"""Extract (goal_state, tactic) pairs from proof datasets.

Takes raw JSONL data from LeanDojo, Lean Workbook, and Goedel-Pset,
and produces training-ready JSONL in BFS-Prover format:

    {"prompt": "{goal_state}:::", "completion": "{tactic}"}

Usage:
    python scripts/extract_tactics.py \
        --input data/raw \
        --output data/processed/train.jsonl \
        --val-output data/processed/val.jsonl \
        --val-split 0.05
"""

import argparse
import json
import logging
import random
from pathlib import Path

from openproof_ml.data.formatting import BANNED_TACTICS, format_training_example

logger = logging.getLogger(__name__)


def extract_leandojo(input_dir: Path) -> list[dict]:
    """Extract (state, tactic) pairs from LeanDojo Benchmark 4 format.

    LeanDojo provides traced tactic data with fields like:
    - state_before: goal state before tactic
    - tactic: the tactic applied
    - state_after: goal state after tactic
    """
    pairs = []
    path = input_dir / "leandojo" / "train.jsonl"
    if not path.exists():
        logger.warning(f"LeanDojo data not found at {path}")
        return pairs

    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            # LeanDojo format varies; try common field names
            state = ex.get("state_before") or ex.get("state") or ex.get("tactic_state", "")
            tactic = ex.get("tactic") or ex.get("action", "")

            if not state or not tactic:
                continue
            if tactic.lower().strip() in BANNED_TACTICS:
                continue

            pairs.append(format_training_example(state.strip(), tactic.strip()))

    logger.info(f"LeanDojo: extracted {len(pairs)} pairs")
    return pairs


def extract_lean_workbook(input_dir: Path) -> list[dict]:
    """Extract from Lean Workbook Plus format."""
    pairs = []
    path = input_dir / "lean_workbook" / "train.jsonl"
    if not path.exists():
        logger.warning(f"Lean Workbook data not found at {path}")
        return pairs

    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            state = ex.get("state_before") or ex.get("tactic_state", "")
            tactic = ex.get("tactic") or ex.get("action", "")

            if not state or not tactic:
                continue
            if tactic.lower().strip() in BANNED_TACTICS:
                continue

            pairs.append(format_training_example(state.strip(), tactic.strip()))

    logger.info(f"Lean Workbook: extracted {len(pairs)} pairs")
    return pairs


def extract_goedel_pset(input_dir: Path) -> list[dict]:
    """Extract from Goedel-Pset-v1-solved format.

    Goedel-Pset contains whole proofs. We split them into individual
    tactic steps. The exact format depends on the dataset version.
    """
    pairs = []
    path = input_dir / "goedel_pset" / "train.jsonl"
    if not path.exists():
        logger.warning(f"Goedel-Pset data not found at {path}")
        return pairs

    with open(path) as f:
        for line in f:
            ex = json.loads(line)

            # Goedel-Pset may have traced tactics or whole proofs
            if "traced_tactics" in ex:
                for step in ex["traced_tactics"]:
                    state = step.get("state_before", "")
                    tactic = step.get("tactic", "")
                    if state and tactic and tactic.lower().strip() not in BANNED_TACTICS:
                        pairs.append(format_training_example(state.strip(), tactic.strip()))
            elif "proof" in ex and "statement" in ex:
                # Whole-proof format -- extract individual tactic lines
                proof = ex["proof"]
                # Simple heuristic: split by newlines, each line is a tactic
                for tactic_line in proof.strip().split("\n"):
                    tactic = tactic_line.strip().lstrip("  ").lstrip("by").strip()
                    if tactic and tactic.lower() not in BANNED_TACTICS:
                        # Without traced state, use the statement as a rough proxy
                        # (proper extraction requires Pantograph replay)
                        state = ex.get("statement", "")
                        if state:
                            pairs.append(format_training_example(state.strip(), tactic))

    logger.info(f"Goedel-Pset: extracted {len(pairs)} pairs")
    return pairs


def deduplicate(pairs: list[dict]) -> list[dict]:
    """Remove exact duplicates."""
    seen = set()
    unique = []
    for p in pairs:
        key = (p["prompt"], p["completion"])
        if key not in seen:
            seen.add(key)
            unique.append(p)
    logger.info(f"Dedup: {len(pairs)} -> {len(unique)} ({len(pairs) - len(unique)} removed)")
    return unique


def main():
    parser = argparse.ArgumentParser(description="Extract tactic training pairs")
    parser.add_argument("--input", required=True, help="Raw data directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--val-output", help="Validation JSONL path")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)

    # Extract from all sources
    all_pairs = []
    all_pairs.extend(extract_leandojo(input_dir))
    all_pairs.extend(extract_lean_workbook(input_dir))
    all_pairs.extend(extract_goedel_pset(input_dir))

    logger.info(f"Total raw pairs: {len(all_pairs)}")

    # Deduplicate
    all_pairs = deduplicate(all_pairs)

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_pairs)

    val_size = int(len(all_pairs) * args.val_split) if args.val_output else 0
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p) + "\n")
    logger.info(f"Wrote {len(train_pairs)} training pairs to {output_path}")

    if args.val_output and val_pairs:
        val_path = Path(args.val_output)
        val_path.parent.mkdir(parents=True, exist_ok=True)
        with open(val_path, "w") as f:
            for p in val_pairs:
                f.write(json.dumps(p) + "\n")
        logger.info(f"Wrote {len(val_pairs)} validation pairs to {val_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
