"""MiniF2F assessment harness.

Runs best-first search on the MiniF2F benchmark and reports pass@k.

Usage:
    python -m openproof_ml.eval.minif2f --config configs/eval_minif2f.yaml
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.formatting import format_prompt, parse_tactic
from ..search.best_first import best_first_search
from ..search.pantograph_client import PantographClient
from ..utils.config import load_config
from .metrics import pass_at_k

logger = logging.getLogger(__name__)


def load_minif2f_problems(problems_dir: str) -> list[dict]:
    """Load MiniF2F problems from Lean 4 source files.

    Each .lean file contains a theorem statement. We extract the type signature.
    """
    problems = []
    problems_path = Path(problems_dir)

    if not problems_path.exists():
        raise FileNotFoundError(
            f"MiniF2F problems not found at {problems_dir}. "
            "Run: git clone https://github.com/yangky11/miniF2F-lean4 data/benchmarks/miniF2F-lean4"
        )

    for lean_file in sorted(problems_path.rglob("*.lean")):
        text = lean_file.read_text()

        for match in re.finditer(
            r"theorem\s+(\w+)\s*(?:\{[^}]*\})?\s*(?:\([^)]*\)\s*)*:\s*(.+?)\s*:=\s*by",
            text,
            re.DOTALL,
        ):
            name = match.group(1)
            type_expr = match.group(2).strip()
            type_expr = re.sub(r"\s+", " ", type_expr)
            problems.append({"name": name, "type_expr": type_expr, "file": str(lean_file)})

    logger.info(f"Loaded {len(problems)} MiniF2F problems from {problems_dir}")
    return problems


def make_propose_fn(model, tokenizer, beam_width: int, temperature: float = 0.8):
    """Create a tactic proposal function."""
    device = next(model.parameters()).device

    def propose_fn(goal_text: str) -> list[str]:
        prompt = format_prompt(goal_text)
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

        tactics = []
        for _ in range(beam_width):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = outputs[0, input_ids.shape[1]:]
            text = tokenizer.decode(response, skip_special_tokens=True)
            tactic = parse_tactic(text)
            if tactic and tactic not in tactics:
                tactics.append(tactic)

        return tactics

    return propose_fn


def run_benchmark(
    model,
    tokenizer,
    pantograph: PantographClient,
    problems: list[dict],
    search_cfg: dict,
    pass_at_k_values: list[int],
    output_dir: Path,
) -> dict:
    """Run the benchmark on all problems.

    For pass@k, we run k independent searches per problem.
    """
    beam_width = search_cfg.get("beam_width", 8)
    max_expansions = search_cfg.get("max_expansions", 200)
    timeout = search_cfg.get("timeout", 120)
    length_penalty = search_cfg.get("length_penalty", 0.1)
    temperature = search_cfg.get("temperature", 0.8)

    max_k = max(pass_at_k_values)
    propose_fn = make_propose_fn(model, tokenizer, beam_width=beam_width, temperature=temperature)

    results = []

    for i, problem in enumerate(problems):
        type_expr = problem["type_expr"]
        name = problem.get("name", f"problem_{i}")

        if not pantograph.is_alive():
            logger.warning("Pantograph crashed, restarting...")
            pantograph.close()
            pantograph.start()

        successes = 0
        attempts = []

        for attempt_num in range(max_k):
            result = best_first_search(
                pantograph, propose_fn, type_expr,
                beam_width=beam_width,
                max_expansions=max_expansions,
                timeout=timeout,
                length_penalty=length_penalty,
            )
            attempts.append({
                "solved": result.solved,
                "tactics": result.tactics if result.solved else [],
                "expansions": result.expansions,
                "elapsed": result.elapsed,
            })
            if result.solved:
                successes += 1

        results.append({
            "name": name,
            "type_expr": type_expr,
            "successes": successes,
            "total_attempts": max_k,
            "attempts": attempts,
        })

        if (i + 1) % 10 == 0:
            solved_so_far = sum(1 for r in results if r["successes"] > 0)
            logger.info(f"Progress: {i + 1}/{len(problems)} | solved: {solved_so_far}")

    # Compute pass@k metrics
    metrics = {}
    for k in pass_at_k_values:
        scores = []
        for r in results:
            scores.append(pass_at_k(r["total_attempts"], r["successes"], k))
        metrics[f"pass@{k}"] = sum(scores) / len(scores)

    total_solved = sum(1 for r in results if r["successes"] > 0)
    metrics["total_solved"] = total_solved
    metrics["total_problems"] = len(problems)
    metrics["solve_rate"] = total_solved / len(problems) if problems else 0

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump({"metrics": metrics, "problems": results}, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    search_cfg = cfg["search"]
    bench_cfg = cfg["eval"]
    panto_cfg = cfg.get("pantograph", {})

    checkpoint = model_cfg["checkpoint"]
    dtype = model_cfg.get("dtype", "bfloat16")
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    logger.info(f"Loading model from {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch_dtype, trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    lean_project = panto_cfg.get("lean_project_path") or "lean"
    repl_path = panto_cfg.get("repl_path")
    pantograph = PantographClient(lean_project, repl_path)
    pantograph.start()
    logger.info("Pantograph ready")

    problems = load_minif2f_problems(bench_cfg["problems_dir"])

    output_dir = Path(bench_cfg.get("output_dir", "outputs/eval"))
    pass_at_k_values = bench_cfg.get("pass_at_k", [1, 8, 32])

    try:
        metrics = run_benchmark(
            model, tokenizer, pantograph, problems,
            search_cfg, pass_at_k_values, output_dir,
        )

        print(f"\n{'='*50}")
        print(f"  MiniF2F Results")
        print(f"{'='*50}")
        print(f"  Model: {checkpoint}")
        print(f"  Problems: {metrics['total_problems']}")
        print(f"  Solved: {metrics['total_solved']}")
        print(f"  Solve rate: {metrics['solve_rate']:.1%}")
        for k in pass_at_k_values:
            print(f"  pass@{k}: {metrics[f'pass@{k}']:.4f}")
        print(f"{'='*50}\n")
        print(f"  Detailed results: {output_dir / 'results.json'}")

    finally:
        pantograph.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
