#!/usr/bin/env python3
"""Launch SFT training. Use with torchrun for multi-GPU:
    torchrun --nproc_per_node=2 scripts/run_sft.py --config configs/sft_qwen35_2b.yaml
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openproof_ml.training.sft import main

if __name__ == "__main__":
    main()
