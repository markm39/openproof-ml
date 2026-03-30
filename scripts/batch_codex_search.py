#!/usr/bin/env python3
"""Batch run Codex tactic search on Goedel-Pset statements.

Creates temp .lean files from statements, runs `openproof tactic-search --file`
in parallel. Each worker gets its own Pantograph instance. Verified pairs are
exported to ~/.openproof/expert-data/positives.jsonl.

Usage:
    python scripts/batch_codex_search.py \
        --input data/raw/goedel_pset/statements.jsonl \
        --workers 8 \
        --limit 1000

Prerequisites:
    - openproof built: cargo install --path crates/openproof-cli
    - Logged in: openproof login
    - Lean project at ../openproof/lean (or set LEAN_PROJECT_PATH)
    - OPENPROOF_TACTIC_PROPOSER=codex in environment
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

OPENPROOF_BIN = os.environ.get("OPENPROOF_BIN", "openproof")


def extract_lean_file(problem: dict) -> str | None:
    """Build a Lean file with sorry from a problem statement."""
    full_proof = problem.get("full_proof", "")
    if full_proof and ":= by" in full_proof:
        # Replace everything after ":= by" with "sorry"
        idx = full_proof.rfind(":= by")
        return full_proof[:idx] + ":= by\n  sorry\n"

    statement = problem.get("formal_statement") or problem.get("statement", "")
    if not statement:
        return None

    # Wrap in a basic Lean file
    if not statement.strip().startswith("import"):
        statement = f"import Mathlib\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n{statement}"

    if ":= by" not in statement:
        return None

    idx = statement.rfind(":= by")
    return statement[:idx] + ":= by\n  sorry\n"


def run_one_problem(args: tuple) -> dict:
    """Run tactic search on one problem. Returns stats."""
    idx, problem, openproof_bin = args

    lean_content = extract_lean_file(problem)
    if lean_content is None:
        return {"idx": idx, "status": "skip", "pairs": 0}

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".lean", prefix=f"expert_{idx:06d}_", delete=False
    ) as f:
        f.write(lean_content)
        lean_path = f.name

    try:
        env = {
            **os.environ,
            "OPENPROOF_TACTIC_PROPOSER": "codex",
            "OPENPROOF_TACTIC_MODEL": os.environ.get("OPENPROOF_TACTIC_MODEL", "gpt-5.4"),
        }

        result = subprocess.run(
            [openproof_bin, "tactic-search", "--file", lean_path],
            capture_output=True,
            text=True,
            timeout=180,  # 3 min per problem
            env=env,
        )

        if result.returncode == 0:
            return {"idx": idx, "status": "ok", "pairs": 1}  # approximate
        else:
            return {"idx": idx, "status": "fail", "pairs": 0}

    except subprocess.TimeoutExpired:
        return {"idx": idx, "status": "timeout", "pairs": 0}
    except Exception as e:
        return {"idx": idx, "status": f"error: {e}", "pairs": 0}
    finally:
        try:
            os.unlink(lean_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Batch Codex tactic search")
    parser.add_argument("--input", required=True, help="JSONL with statements")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--limit", type=int, help="Max problems")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N")
    parser.add_argument("--openproof", default=OPENPROOF_BIN, help="openproof binary path")
    args = parser.parse_args()

    # Load problems
    problems = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if i < args.offset:
                continue
            if args.limit and len(problems) >= args.limit:
                break
            problems.append(json.loads(line))

    logger.info(f"Loaded {len(problems)} problems, using {args.workers} workers")

    # Check openproof is available
    try:
        subprocess.run([args.openproof, "--help"], capture_output=True, timeout=5)
    except FileNotFoundError:
        logger.error(f"openproof binary not found at {args.openproof}")
        logger.error("Build it: cd ../openproof && cargo install --path crates/openproof-cli")
        return

    export_dir = Path.home() / ".openproof" / "expert-data"
    positives_path = export_dir / "positives.jsonl"
    initial_count = 0
    if positives_path.exists():
        initial_count = sum(1 for _ in open(positives_path))
    logger.info(f"Starting with {initial_count} existing pairs in {positives_path}")

    # Build work items
    work = [(i + args.offset, p, args.openproof) for i, p in enumerate(problems)]

    ok = 0
    fail = 0
    skip = 0
    timeout = 0
    start_time = time.monotonic()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one_problem, w): w[0] for w in work}
        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "ok":
                ok += 1
            elif result["status"] == "skip":
                skip += 1
            elif result["status"] == "timeout":
                timeout += 1
            else:
                fail += 1

            total_done = ok + fail + skip + timeout
            if total_done % 10 == 0:
                elapsed = time.monotonic() - start_time
                rate = total_done / elapsed if elapsed > 0 else 0
                # Count new pairs
                new_count = 0
                if positives_path.exists():
                    new_count = sum(1 for _ in open(positives_path)) - initial_count
                logger.info(
                    f"Progress: {total_done}/{len(problems)} "
                    f"ok={ok} fail={fail} skip={skip} timeout={timeout} "
                    f"new_pairs={new_count} rate={rate:.1f}/s"
                )

    # Final count
    final_count = 0
    if positives_path.exists():
        final_count = sum(1 for _ in open(positives_path))
    new_pairs = final_count - initial_count

    elapsed = time.monotonic() - start_time
    logger.info(f"\n=== Done in {elapsed:.0f}s ===")
    logger.info(f"Problems: {len(problems)} (ok={ok} fail={fail} skip={skip} timeout={timeout})")
    logger.info(f"New verified pairs: {new_pairs}")
    logger.info(f"Total pairs in {positives_path}: {final_count}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
