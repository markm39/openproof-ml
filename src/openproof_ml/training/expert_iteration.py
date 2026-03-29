"""Expert iteration with V-STaR for tactic prediction.

Self-play loop:
  1. Run best-first search with current model on unsolved problems
  2. Collect successful proofs (positive) and failed attempts (negative)
  3. SFT on positive pairs to reinforce solutions
  4. DPO on (positive, negative) pairs to learn from contrast
  5. Repeat for N rounds

Usage:
    python -m openproof_ml.training.expert_iteration --config configs/expert_iter.yaml
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from trl import DPOConfig, DPOTrainer

from ..data.formatting import SEPARATOR, format_prompt, parse_tactic
from ..search.best_first import SearchResult, best_first_search
from ..search.pantograph_client import PantographClient
from ..utils.config import load_config

logger = logging.getLogger(__name__)


def load_problems(problems_dir: str) -> list[dict]:
    """Load unsolved theorem statements from JSONL files."""
    problems = []
    problems_path = Path(problems_dir)

    if problems_path.is_file():
        with open(problems_path) as f:
            for line in f:
                data = json.loads(line.strip())
                expr = data.get("type_expr") or data.get("goal_state") or data.get("statement")
                if expr:
                    problems.append({"type_expr": expr, "name": data.get("name", "")})
    elif problems_path.is_dir():
        for jsonl_file in sorted(problems_path.glob("*.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    data = json.loads(line.strip())
                    expr = data.get("type_expr") or data.get("goal_state") or data.get("statement")
                    if expr:
                        problems.append({"type_expr": expr, "name": data.get("name", "")})

    logger.info(f"Loaded {len(problems)} problems from {problems_dir}")
    return problems


def make_propose_fn(model, tokenizer, beam_width: int = 16, temperature: float = 0.8):
    """Create a tactic proposal function from the model."""
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


def run_search_round(
    model,
    tokenizer,
    pantograph: PantographClient,
    problems: list[dict],
    search_cfg: dict,
) -> tuple[list[dict], list[dict]]:
    """Run BFS on all problems, collect positive and negative pairs.

    Returns (positives, negatives) where each is a list of
    {prompt, chosen/rejected} dicts.
    """
    beam_width = search_cfg.get("beam_width", 16)
    max_expansions = search_cfg.get("max_expansions", 400)
    timeout = search_cfg.get("timeout", 120)
    length_penalty = search_cfg.get("length_penalty", 0.1)

    propose_fn = make_propose_fn(model, tokenizer, beam_width=beam_width)

    positives = []
    negatives = []
    solved_count = 0

    for i, problem in enumerate(problems):
        type_expr = problem["type_expr"]

        if not pantograph.is_alive():
            logger.warning("Pantograph crashed, restarting...")
            pantograph.close()
            pantograph.start()

        result = best_first_search(
            pantograph, propose_fn, type_expr,
            beam_width=beam_width,
            max_expansions=max_expansions,
            timeout=timeout,
            length_penalty=length_penalty,
        )

        if result.solved:
            solved_count += 1
            # Each tactic step is a positive training example
            # Re-trace the proof to get (state, tactic) pairs
            state_id = pantograph.start_goal(type_expr)
            if state_id is not None:
                current_goals = [type_expr]
                for tactic in result.tactics:
                    goal_text = current_goals[0] if current_goals else type_expr
                    positives.append({
                        "prompt": format_prompt(goal_text),
                        "chosen": tactic,
                    })
                    # Apply tactic to advance state
                    tac_result = pantograph.try_tactic(state_id, 0, tactic)
                    if tac_result.success and tac_result.new_state_id is not None:
                        pantograph.delete_goal(state_id)
                        state_id = tac_result.new_state_id
                        current_goals = tac_result.remaining_goals
                    else:
                        break
                pantograph.delete_goal(state_id)
        else:
            # Failed: use the goal state as a negative example prompt
            # The "rejected" tactic is whatever the model tried that failed
            negatives.append({
                "prompt": format_prompt(type_expr),
                "rejected": "sorry",  # placeholder -- model couldn't solve it
            })

        if (i + 1) % 100 == 0:
            logger.info(f"Search progress: {i + 1}/{len(problems)} solved={solved_count}")

    logger.info(f"Round complete: {solved_count}/{len(problems)} solved, {len(positives)} positive pairs")
    return positives, negatives


def filter_beam1_solvable(
    model, tokenizer, pantograph: PantographClient,
    problems: list[dict], search_cfg: dict,
) -> list[dict]:
    """Remove problems solvable with beam_width=1 (too easy)."""
    propose_fn = make_propose_fn(model, tokenizer, beam_width=1, temperature=0.0)
    hard_problems = []

    for problem in problems:
        result = best_first_search(
            pantograph, propose_fn, problem["type_expr"],
            beam_width=1, max_expansions=50, timeout=30,
        )
        if not result.solved:
            hard_problems.append(problem)

    logger.info(f"Filtered: {len(problems)} -> {len(hard_problems)} hard problems")
    return hard_problems


def train_sft_on_positives(
    model, tokenizer, positives: list[dict],
    output_dir: Path, learning_rate: float = 1e-5,
):
    """SFT on successful proof trajectories."""
    if not positives:
        logger.warning("No positive examples to train on")
        return

    # Build HuggingFace dataset
    texts = [p["prompt"] + p["chosen"] for p in positives]
    prompts = [p["prompt"] for p in positives]

    encodings = tokenizer(texts, truncation=True, max_length=2048, padding=False)

    # Mask prompt tokens
    labels = []
    for prompt, ids in zip(prompts, encodings["input_ids"]):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        label = list(ids)
        label[: len(prompt_ids)] = [-100] * len(prompt_ids)
        labels.append(label)

    encodings["labels"] = labels
    dataset = HFDataset.from_dict(dict(encodings))

    training_args = TrainingArguments(
        output_dir=str(output_dir / "sft"),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        seed=42,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=dataset, tokenizer=tokenizer,
    )
    trainer.train()
    logger.info("SFT on positives complete")


def train_dpo_on_pairs(
    model, tokenizer, positives: list[dict], negatives: list[dict],
    output_dir: Path, learning_rate: float = 5e-6,
):
    """DPO on (positive, negative) pairs (V-STaR)."""
    # Match positives with negatives by prompt where possible
    # Fall back to pairing with "sorry" as rejected
    pairs = []
    neg_by_prompt = {}
    for neg in negatives:
        neg_by_prompt[neg["prompt"]] = neg.get("rejected", "sorry")

    for pos in positives:
        rejected = neg_by_prompt.get(pos["prompt"], "sorry")
        pairs.append({
            "prompt": pos["prompt"],
            "chosen": pos["chosen"],
            "rejected": rejected,
        })

    if not pairs:
        logger.warning("No DPO pairs to train on")
        return

    dataset = HFDataset.from_list(pairs)

    dpo_config = DPOConfig(
        output_dir=str(output_dir / "dpo"),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        max_length=2048,
        max_prompt_length=1024,
        report_to="none",
    )

    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    dpo_trainer.train()
    logger.info("DPO training complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    search_cfg = cfg["search"]
    data_cfg = cfg["data"]
    filter_cfg = cfg.get("filtering", {})
    panto_cfg = cfg.get("pantograph", {})

    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg:
        import wandb
        wandb.init(project=wandb_cfg.get("project"), name=wandb_cfg.get("name"))

    # Load model
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

    # Load problems
    problems = load_problems(data_cfg["problems_dir"])

    # Start Pantograph
    lean_project = panto_cfg.get("lean_project_path") or "lean"
    repl_path = panto_cfg.get("repl_path")
    pantograph = PantographClient(lean_project, repl_path)
    pantograph.start()
    logger.info("Pantograph ready")

    output_dir = Path(data_cfg.get("output_dir", "data/expert_iter"))
    output_dir.mkdir(parents=True, exist_ok=True)
    max_rounds = data_cfg.get("max_rounds", 3)

    try:
        for round_num in range(1, max_rounds + 1):
            logger.info(f"=== Expert Iteration Round {round_num}/{max_rounds} ===")

            # Filter easy problems (beam=1 solvable)
            if filter_cfg.get("remove_beam1_solvable", True) and round_num == 1:
                problems = filter_beam1_solvable(model, tokenizer, pantograph, problems, search_cfg)

            # Run search
            positives, negatives = run_search_round(
                model, tokenizer, pantograph, problems, search_cfg,
            )

            # Save trajectories
            round_dir = output_dir / f"round_{round_num}"
            round_dir.mkdir(parents=True, exist_ok=True)
            with open(round_dir / "positives.jsonl", "w") as f:
                for p in positives:
                    f.write(json.dumps(p) + "\n")
            with open(round_dir / "negatives.jsonl", "w") as f:
                for n in negatives:
                    f.write(json.dumps(n) + "\n")

            if not positives:
                logger.warning(f"Round {round_num}: no solutions found, stopping")
                break

            # SFT on positives
            logger.info(f"Round {round_num}: SFT on {len(positives)} positive pairs")
            train_sft_on_positives(model, tokenizer, positives, round_dir)

            # DPO on (positive, negative) pairs
            logger.info(f"Round {round_num}: DPO on {len(positives)} pairs")
            train_dpo_on_pairs(model, tokenizer, positives, negatives, round_dir)

            # Save round checkpoint
            ckpt_dir = output_dir / f"checkpoint-round-{round_num}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Round {round_num} checkpoint saved: {ckpt_dir}")

        # Save final model
        final_dir = output_dir / "final"
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        logger.info(f"Expert iteration complete. Final model: {final_dir}")

    finally:
        pantograph.close()
        if wandb_cfg:
            wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
