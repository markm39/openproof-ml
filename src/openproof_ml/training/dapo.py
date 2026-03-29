"""DAPO (Dynamic sampling Asymmetric PPO) RL training for tactic prediction.

Trains the model with per-tactic rewards from Pantograph:
  - goal_closed: 1.0 (tactic closes a goal)
  - state_changed: 0.5 (tactic succeeds, changes goal state)
  - error: 0.0 (tactic fails)

Key DAPO features:
  - Asymmetric clipping (eps_low < eps_high) for exploration
  - No KL penalty (Pantograph is a perfect verifier)
  - Dynamic sampling (skip all-pass/all-fail prompts)
  - Dr. GRPO length normalization

Usage:
    python -m openproof_ml.training.dapo --config configs/dapo.yaml
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from ..data.formatting import SEPARATOR, format_prompt, parse_tactic
from ..search.pantograph_client import PantographClient, TacticResult
from ..utils.config import load_config

logger = logging.getLogger(__name__)


class PromptDataset(Dataset):
    """Dataset of theorem goal states for RL rollouts."""

    def __init__(self, prompts_file: str):
        self.prompts: list[str] = []
        with open(prompts_file) as f:
            for line in f:
                data = json.loads(line.strip())
                goal = data.get("goal_state") or data.get("prompt", "").rstrip(SEPARATOR)
                if goal:
                    self.prompts.append(goal)
        logger.info(f"Loaded {len(self.prompts)} RL prompts")

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]


def compute_tactic_reward(
    pantograph: PantographClient,
    state_id: int,
    goal_id: int,
    tactic: str,
    reward_cfg: dict,
) -> tuple[float, TacticResult | None]:
    """Apply a tactic and compute the per-tactic reward.

    Returns (reward, tactic_result). tactic_result is None if the tactic failed.
    """
    result = pantograph.try_tactic(state_id, goal_id, tactic)

    if not result.success:
        return reward_cfg.get("error", 0.0), None

    if not result.remaining_goals:
        return reward_cfg.get("goal_closed", 1.0), result

    return reward_cfg.get("state_changed", 0.5), result


def generate_rollouts(
    model: torch.nn.Module,
    tokenizer,
    pantograph: PantographClient,
    goal_state: str,
    num_rollouts: int,
    reward_cfg: dict,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
) -> list[dict]:
    """Generate multiple tactic rollouts for a single goal state and score them.

    Each rollout generates a single tactic and scores it via Pantograph.
    Returns list of dicts with prompt_ids, response_ids, reward, tactic.
    """
    prompt_text = format_prompt(goal_state)
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    prompt_ids = prompt_enc["input_ids"].to(model.device)
    prompt_len = prompt_ids.shape[1]

    rollouts = []

    for _ in range(num_rollouts):
        with torch.no_grad():
            outputs = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response_ids = outputs[0, prompt_len:]
        tactic_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        tactic = parse_tactic(tactic_text)

        if tactic is None:
            rollouts.append({
                "prompt_ids": prompt_ids[0],
                "response_ids": response_ids,
                "reward": reward_cfg.get("error", 0.0),
                "tactic": tactic_text,
            })
            continue

        # Score via Pantograph
        state_id = pantograph.start_goal(goal_state)
        if state_id is None:
            rollouts.append({
                "prompt_ids": prompt_ids[0],
                "response_ids": response_ids,
                "reward": reward_cfg.get("error", 0.0),
                "tactic": tactic,
            })
            continue

        reward, _ = compute_tactic_reward(pantograph, state_id, 0, tactic, reward_cfg)
        pantograph.delete_goal(state_id)

        rollouts.append({
            "prompt_ids": prompt_ids[0],
            "response_ids": response_ids,
            "reward": reward,
            "tactic": tactic,
        })

    return rollouts


def compute_log_probs(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log probabilities for a response given a prompt."""
    input_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids).logits

    prompt_len = len(prompt_ids)
    response_logits = logits[0, prompt_len - 1 : -1]
    response_labels = response_ids

    log_probs = torch.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(1, response_labels.unsqueeze(1)).squeeze(1)
    return token_log_probs


def dapo_loss(
    model: torch.nn.Module,
    ref_log_probs: torch.Tensor,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    advantage: float,
    eps_low: float,
    eps_high: float,
    length_normalize: bool,
) -> torch.Tensor:
    """Compute the DAPO policy gradient loss for a single rollout.

    Asymmetric clipping:
      - Positive advantages: clip ratio at (1 + eps_high)
      - Negative advantages: clip ratio at (1 - eps_low)

    Dr. GRPO: advantages divided by response length.
    """
    input_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(model.device)
    logits = model(input_ids).logits

    prompt_len = len(prompt_ids)
    response_logits = logits[0, prompt_len - 1 : -1]
    response_labels = response_ids.to(model.device)

    log_probs = torch.log_softmax(response_logits, dim=-1)
    current_log_probs = log_probs.gather(1, response_labels.unsqueeze(1)).squeeze(1)

    ratio = torch.exp(current_log_probs - ref_log_probs.to(model.device))

    if length_normalize and len(response_ids) > 0:
        effective_advantage = advantage / len(response_ids)
    else:
        effective_advantage = advantage

    if effective_advantage >= 0:
        clipped_ratio = torch.clamp(ratio, max=1.0 + eps_high)
    else:
        clipped_ratio = torch.clamp(ratio, min=1.0 - eps_low)

    surrogate = ratio * effective_advantage
    clipped_surrogate = clipped_ratio * effective_advantage
    loss = -torch.min(surrogate, clipped_surrogate).mean()

    return loss


def should_skip_prompt(rollouts: list[dict]) -> bool:
    """Dynamic sampling: skip if all rollouts pass or all fail (no gradient signal)."""
    rewards = [r["reward"] for r in rollouts]
    return all(r > 0 for r in rewards) or all(r == 0 for r in rewards)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    dapo_cfg = cfg["dapo"]
    train_cfg = cfg["training"]
    panto_cfg = cfg.get("pantograph", {})
    data_cfg = cfg["data"]

    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg:
        import wandb
        wandb.init(project=wandb_cfg.get("project"), name=wandb_cfg.get("name"))

    # Load model from SFT checkpoint
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

    if model_cfg.get("use_lora", False):
        lora_config = LoraConfig(
            r=model_cfg.get("lora_rank", 64),
            lora_alpha=model_cfg.get("lora_alpha", 128),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get("learning_rate", 5e-6), weight_decay=0.01)

    prompt_dataset = PromptDataset(data_cfg["prompts_file"])
    num_prompts = len(prompt_dataset)
    num_epochs = train_cfg.get("num_epochs", 2)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 4)
    total_steps = (num_prompts * num_epochs) // grad_accum
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.05))

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # DAPO hyperparameters
    num_rollouts = dapo_cfg.get("num_rollouts", 32)
    eps_low = dapo_cfg.get("eps_low", 0.1)
    eps_high = dapo_cfg.get("eps_high", 0.2)
    length_normalize = dapo_cfg.get("length_normalize", True)
    dynamic_sampling = dapo_cfg.get("dynamic_sampling", True)
    reward_cfg = dapo_cfg.get("rewards", {"goal_closed": 1.0, "state_changed": 0.5, "error": 0.0})

    # Start Pantograph
    lean_project = panto_cfg.get("lean_project_path") or "lean"
    repl_path = panto_cfg.get("repl_path")
    logger.info("Starting Pantograph REPL...")
    pantograph = PantographClient(lean_project, repl_path)
    pantograph.start()
    logger.info("Pantograph ready")

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_steps = train_cfg.get("save_steps", 100)

    global_step = 0
    model.train()

    try:
        for epoch in range(num_epochs):
            indices = list(range(num_prompts))
            random.shuffle(indices)

            accum_loss = 0.0
            accum_count = 0
            skipped = 0
            total_reward = 0.0
            total_rollouts = 0

            for prompt_idx, idx in enumerate(indices):
                goal_state = prompt_dataset[idx]

                if not pantograph.is_alive():
                    logger.warning("Pantograph crashed, restarting...")
                    pantograph.close()
                    pantograph = PantographClient(lean_project, repl_path)
                    pantograph.start()

                # Generate rollouts (model in eval mode for generation)
                model.eval()
                rollouts = generate_rollouts(
                    model, tokenizer, pantograph, goal_state,
                    num_rollouts=num_rollouts, reward_cfg=reward_cfg,
                )
                model.train()

                if dynamic_sampling and should_skip_prompt(rollouts):
                    skipped += 1
                    continue

                # Compute normalized advantages
                rewards = [r["reward"] for r in rollouts]
                mean_reward = sum(rewards) / len(rewards)
                std_reward = max((sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5, 1e-8)
                total_reward += sum(rewards)
                total_rollouts += len(rewards)

                # Freeze reference log probs
                model.eval()
                ref_log_probs_list = []
                for rollout in rollouts:
                    ref_lp = compute_log_probs(model, rollout["prompt_ids"], rollout["response_ids"])
                    ref_log_probs_list.append(ref_lp.detach())
                model.train()

                # Accumulate DAPO loss
                for rollout, ref_lp in zip(rollouts, ref_log_probs_list):
                    advantage = (rollout["reward"] - mean_reward) / std_reward
                    loss = dapo_loss(
                        model, ref_lp, rollout["prompt_ids"], rollout["response_ids"],
                        advantage=advantage, eps_low=eps_low, eps_high=eps_high,
                        length_normalize=length_normalize,
                    )
                    loss = loss / (len(rollouts) * grad_accum)
                    loss.backward()
                    accum_loss += loss.item() * len(rollouts) * grad_accum
                    accum_count += 1

                # Optimizer step
                if (prompt_idx + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("max_grad_norm", 1.0))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    avg_loss = accum_loss / max(accum_count, 1)
                    avg_reward = total_reward / max(total_rollouts, 1)

                    if global_step % train_cfg.get("logging_steps", 1) == 0:
                        logger.info(
                            f"step={global_step} loss={avg_loss:.4f} "
                            f"avg_reward={avg_reward:.3f} skipped={skipped} "
                            f"lr={scheduler.get_last_lr()[0]:.2e}"
                        )
                        if wandb_cfg:
                            wandb.log({
                                "loss": avg_loss, "avg_reward": avg_reward,
                                "skipped_prompts": skipped,
                                "lr": scheduler.get_last_lr()[0],
                                "global_step": global_step,
                            })

                    accum_loss = 0.0
                    accum_count = 0

                    if global_step % save_steps == 0:
                        ckpt_dir = output_dir / f"checkpoint-{global_step}"
                        model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        logger.info(f"Saved checkpoint: {ckpt_dir}")

            logger.info(f"Epoch {epoch + 1}/{num_epochs} complete. steps={global_step} skipped={skipped}/{num_prompts}")

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Training complete. Model saved to {output_dir}")

    finally:
        pantograph.close()
        if wandb_cfg:
            wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
