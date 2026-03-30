#!/usr/bin/env python3
"""Generate (state, tactic) training data using Codex + Pantograph verification.

Sends goal states to Codex (GPT-5.4) for tactic proposals, verifies each
with Pantograph, and exports verified (state, tactic) pairs for SFT training.

Massively parallelizable: each worker spawns its own Pantograph instance.

Usage:
    # Run on Goedel-Pset statements (default)
    python scripts/codex_expert_play.py \
        --input data/raw/goedel_pset/statements.jsonl \
        --output data/expert/codex_pairs.jsonl \
        --workers 8 \
        --limit 10000

    # Requires: OPENAI_API_KEY or CODEX_API_KEY env var
    # Requires: Lean + Pantograph built (make setup-lean)
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a Lean 4 tactic expert. Given a goal state, return a JSON object "
    '{"tactics":["tactic1","tactic2",...]} with up to 8 single-line Lean 4 tactics '
    "that could close or make progress on the goal. No markdown, no explanation. "
    "Never use sorry, admit, or native_decide."
)

BANNED_TACTICS = {"sorry", "admit", "native_decide"}


def call_codex(goal_state: str, model: str = "gpt-5.4", api_key: str = None) -> list[str]:
    """Call Codex/OpenAI API with a goal state, return tactic suggestions."""
    import httpx

    key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY or CODEX_API_KEY")

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Goal state:\n{goal_state}"},
            ],
            "temperature": 0.8,
            "max_tokens": 512,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return parse_tactics(content)


def parse_tactics(response: str) -> list[str]:
    """Parse tactics from Codex JSON response."""
    import re

    # Try direct JSON parse
    try:
        data = json.loads(response)
        if isinstance(data, dict) and "tactics" in data:
            return [t.strip() for t in data["tactics"] if t.strip()]
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown
    cleaned = response.strip().strip("`").strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "tactics" in data:
            return [t.strip() for t in data["tactics"] if t.strip()]
    except json.JSONDecodeError:
        pass

    # Try finding JSON object in text
    match = re.search(r'\{[^{}]*"tactics"\s*:\s*\[.*?\]\s*\}', response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return [t.strip() for t in data["tactics"] if t.strip()]
        except json.JSONDecodeError:
            pass

    return []


def filter_tactic(tactic: str) -> bool:
    """Return True if tactic is safe to use."""
    lower = tactic.lower().strip()
    for banned in BANNED_TACTICS:
        if lower == banned or lower.startswith(f"{banned} ") or lower.startswith(f"{banned};"):
            return False
    if "?_" in tactic:
        return False
    return True


def extract_type_from_statement(statement: str) -> str | None:
    """Extract forall type expression from a Lean theorem statement."""
    import re

    match = re.match(r"(?:theorem|lemma|def)\s+\S+\s*(.*)", statement, re.DOTALL)
    if not match:
        return None

    rest = match.group(1).strip()
    for suffix in [":= by sorry", ":= by", ":= sorry", ":="]:
        idx = rest.rfind(suffix)
        if idx != -1:
            rest = rest[:idx].strip()
            break

    # Find last top-level colon
    depth = 0
    colon_pos = None
    for i in range(len(rest) - 1, -1, -1):
        c = rest[i]
        if c in ")]}":
            depth += 1
        elif c in "([{":
            depth -= 1
        elif c == ":" and depth == 0 and i + 1 < len(rest) and rest[i + 1] != "=":
            colon_pos = i
            break

    if colon_pos is None:
        return None

    params = rest[:colon_pos].strip()
    conclusion = rest[colon_pos + 1:].strip()
    if not params:
        return conclusion
    return f"forall {params}, {conclusion}"


def worker_process(args: tuple) -> list[dict]:
    """Worker: process a chunk of problems with own Pantograph + Codex calls."""
    worker_id, problems, lean_project, repl_path, model, api_key = args

    from openproof_ml.search.pantograph_client import PantographClient

    pairs = []
    solved = 0
    failed = 0

    pg = PantographClient(lean_project, repl_path)
    pg.start()
    logger.info(f"Worker {worker_id}: started, {len(problems)} problems")

    for i, problem in enumerate(problems):
        # Extract the theorem statement
        statement = problem.get("formal_statement") or problem.get("statement", "")
        full_proof = problem.get("full_proof", "")

        if full_proof:
            # Parse theorem from full_proof
            by_idx = full_proof.rfind(":= by")
            if by_idx == -1:
                failed += 1
                continue
            before = full_proof[:by_idx]
            for kw in ["theorem ", "lemma ", "def "]:
                kw_idx = before.rfind(kw)
                if kw_idx != -1:
                    statement = full_proof[kw_idx:by_idx].strip()
                    break

        type_expr = extract_type_from_statement(statement)
        if not type_expr:
            failed += 1
            continue

        # Start goal in Pantograph
        if not pg.is_alive():
            pg.close()
            pg = PantographClient(lean_project, repl_path)
            pg.start()

        state_id = pg.start_goal(type_expr)
        if state_id is None:
            failed += 1
            continue

        # Get tactics from Codex
        try:
            tactics = call_codex(type_expr, model=model, api_key=api_key)
        except Exception as e:
            failed += 1
            pg.delete_goal(state_id)
            if failed <= 5:
                logger.warning(f"W{worker_id}[{i}] Codex error: {e}")
            continue

        tactics = [t for t in tactics if filter_tactic(t)]

        # Try each tactic with Pantograph
        current_state = state_id
        current_goal = type_expr
        proof_tactics = []
        allocated = [state_id]

        for tactic in tactics:
            result = pg.try_tactic(current_state, 0, tactic)
            if result.success and result.new_state_id is not None:
                # Record the verified pair
                pairs.append({
                    "prompt": f"{current_goal}:::",
                    "completion": tactic,
                })
                proof_tactics.append(tactic)
                allocated.append(result.new_state_id)

                if not result.remaining_goals:
                    # Proof complete!
                    solved += 1
                    break
                current_state = result.new_state_id
                current_goal = result.remaining_goals[0]

        for sid in allocated:
            try:
                pg.delete_goal(sid)
            except Exception:
                pass

        if (i + 1) % 100 == 0:
            logger.info(
                f"W{worker_id}: {i+1}/{len(problems)} "
                f"solved={solved} failed={failed} pairs={len(pairs)}"
            )

    pg.close()
    logger.info(
        f"Worker {worker_id} done: solved={solved} failed={failed} pairs={len(pairs)}"
    )
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate training data via Codex + Pantograph")
    parser.add_argument("--input", required=True, help="JSONL with theorem statements")
    parser.add_argument("--output", required=True, help="Output JSONL for verified pairs")
    parser.add_argument("--lean-project", default="lean", help="Lean project path")
    parser.add_argument("--pantograph", default="vendor/Pantograph/.lake/build/bin/repl")
    parser.add_argument("--model", default="gpt-5.4", help="Codex model name")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--limit", type=int, help="Max problems to process")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N problems")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY")
    if not api_key:
        logger.error("Set OPENAI_API_KEY or CODEX_API_KEY")
        return

    # Load problems
    problems = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if i < args.offset:
                continue
            if args.limit and len(problems) >= args.limit:
                break
            problems.append(json.loads(line))

    logger.info(f"Loaded {len(problems)} problems")

    if args.workers <= 1:
        all_pairs = worker_process(
            (0, problems, args.lean_project, args.pantograph, args.model, api_key)
        )
    else:
        # Shard across workers
        chunk_size = (len(problems) + args.workers - 1) // args.workers
        chunks = []
        for w in range(args.workers):
            start = w * chunk_size
            end = min(start + chunk_size, len(problems))
            if start < len(problems):
                chunks.append(
                    (w, problems[start:end], args.lean_project, args.pantograph, args.model, api_key)
                )

        logger.info(f"Spawning {len(chunks)} workers")
        all_pairs = []
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(worker_process, chunk) for chunk in chunks]
            for future in as_completed(futures):
                all_pairs.extend(future.result())

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info(f"Wrote {len(all_pairs)} verified pairs to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
