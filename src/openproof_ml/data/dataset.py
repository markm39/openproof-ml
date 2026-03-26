"""Dataset classes for tactic prediction training."""

import json
from pathlib import Path

from torch.utils.data import Dataset


class TacticDataset(Dataset):
    """Dataset of (goal_state, tactic) pairs for SFT training.

    Each example is a JSONL line with 'prompt' and 'completion' fields:
        {"prompt": "a b : Nat\\n|- a + b = b + a:::", "completion": "omega"}
    """

    def __init__(self, path: str | Path, max_examples: int | None = None):
        self.examples = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                ex = json.loads(line)
                self.examples.append(ex)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.examples[idx]

    @staticmethod
    def collate_for_sft(examples: list[dict], tokenizer, max_length: int = 2048):
        """Collate examples into a batch for SFT training.

        Concatenates prompt + completion, with labels masked on the prompt portion.
        """
        texts = [ex["prompt"] + ex["completion"] for ex in examples]
        prompts = [ex["prompt"] for ex in examples]

        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Create labels: -100 for prompt tokens (don't compute loss on them)
        labels = encodings["input_ids"].clone()
        for i, prompt in enumerate(prompts):
            prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            labels[i, :prompt_len] = -100

        encodings["labels"] = labels
        return encodings
