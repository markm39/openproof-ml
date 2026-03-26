"""Prompt formatting for tactic prediction models.

This module defines the contract between training and inference.
The format must match openproof's OllamaProposer (ollama.rs) exactly:

    Input:  {goal_state}:::
    Output: {tactic}

The model is a raw completion model (no chat template).
"""

SEPARATOR = ":::"

# Tactics that should never appear in training data or model output.
BANNED_TACTICS = frozenset({"sorry", "admit", "native_decide"})

# Substrings that indicate buggy tactic output.
BANNED_SUBSTRINGS = ("?_",)


def format_prompt(goal_state: str) -> str:
    """Format a goal state as a model input prompt.

    Args:
        goal_state: Pretty-printed Lean goal state (Pantograph target.pp format).

    Returns:
        Prompt string ready for tokenization.
    """
    return f"{goal_state}{SEPARATOR}"


def format_training_example(goal_state: str, tactic: str) -> dict[str, str]:
    """Format a (state, tactic) pair as a training example.

    Returns:
        Dict with 'prompt' and 'completion' fields for SFT.
    """
    return {
        "prompt": format_prompt(goal_state),
        "completion": tactic,
    }


def parse_tactic(model_output: str) -> str | None:
    """Parse a tactic from raw model output.

    Handles common issues: trailing separators, multi-line output,
    banned tactics, whitespace.

    Args:
        model_output: Raw string from the model.

    Returns:
        Cleaned tactic string, or None if the tactic should be filtered.
    """
    # Take first line only (model may generate multi-line)
    tactic = model_output.split("\n")[0].strip()

    # Strip trailing separator (model sometimes echoes it)
    tactic = tactic.rstrip(SEPARATOR).strip()

    if not tactic:
        return None

    # Check banned tactics (whole word at start)
    lower = tactic.lower()
    for banned in BANNED_TACTICS:
        if lower == banned or lower.startswith(f"{banned} ") or lower.startswith(f"{banned};"):
            return None

    # Check banned substrings
    for sub in BANNED_SUBSTRINGS:
        if sub in tactic:
            return None

    return tactic
