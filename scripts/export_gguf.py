#!/usr/bin/env python3
"""Export a trained checkpoint to GGUF format and register with ollama.

Steps:
  1. Merge LoRA weights into base model (if applicable)
  2. Convert to GGUF Q4_K_M quantization (~1.5GB for 2B model)
  3. Create ollama Modelfile
  4. Register with ollama

Usage:
    python scripts/export_gguf.py --config configs/eval_minif2f.yaml
    python scripts/export_gguf.py --checkpoint checkpoints/dapo-qwen35-2b --quantization q4_k_m
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_lora(checkpoint_dir: Path, output_dir: Path) -> Path:
    """Merge LoRA adapter weights into the base model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Checking for LoRA adapter in {checkpoint_dir}")

    adapter_config = checkpoint_dir / "adapter_config.json"
    if not adapter_config.exists():
        logger.info("No LoRA adapter found, using checkpoint directly")
        return checkpoint_dir

    logger.info("Merging LoRA weights into base model...")
    import json

    with open(adapter_config) as f:
        config = json.load(f)
    base_model_name = config.get("base_model_name_or_path", "")

    if not base_model_name:
        raise ValueError("Cannot determine base model from adapter_config.json")

    import torch

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Merged model saved to {output_dir}")
    return output_dir


def convert_to_gguf(model_dir: Path, output_path: Path, quantization: str = "q4_k_m"):
    """Convert a HuggingFace model to GGUF format.

    Requires llama.cpp's convert script or llama-cpp-python.
    """
    logger.info(f"Converting to GGUF ({quantization})...")

    # Try llama.cpp convert_hf_to_gguf.py first
    llama_cpp_convert = shutil.which("convert_hf_to_gguf.py")
    if llama_cpp_convert is None:
        # Try common locations
        for candidate in [
            Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
            Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
        ]:
            if candidate.exists():
                llama_cpp_convert = str(candidate)
                break

    if llama_cpp_convert:
        # Use llama.cpp native converter
        fp16_path = output_path.with_suffix(".fp16.gguf")
        subprocess.run(
            [sys.executable, llama_cpp_convert, str(model_dir), "--outfile", str(fp16_path), "--outtype", "f16"],
            check=True,
        )

        # Quantize
        quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
        if quantize_bin:
            subprocess.run(
                [quantize_bin, str(fp16_path), str(output_path), quantization.upper()],
                check=True,
            )
            fp16_path.unlink()  # Clean up fp16 intermediate
        else:
            # No quantize binary, keep fp16
            fp16_path.rename(output_path)
            logger.warning("llama-quantize not found, keeping fp16 GGUF (larger file)")
    else:
        # Fallback: try llama-cpp-python
        try:
            from llama_cpp import llama_model_quantize

            logger.info("Using llama-cpp-python for conversion")
            # This is a simplified path; in practice you'd use the full pipeline
            raise NotImplementedError("Use llama.cpp convert_hf_to_gguf.py for best results")
        except ImportError:
            raise RuntimeError(
                "Neither llama.cpp nor llama-cpp-python found. Install one:\n"
                "  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make\n"
                "  OR: pip install llama-cpp-python"
            )

    logger.info(f"GGUF saved to {output_path}")
    return output_path


def create_ollama_modelfile(gguf_path: Path, model_name: str) -> Path:
    """Create an ollama Modelfile for the tactic model."""
    modelfile_path = gguf_path.parent / "Modelfile"

    content = f"""FROM {gguf_path.resolve()}

# Tactic prediction model for OpenProof
# Input format: {{goal_state}}:::
# Output: single tactic

PARAMETER temperature 0.8
PARAMETER top_p 0.95
PARAMETER stop "\\n"
PARAMETER num_predict 256

TEMPLATE "{{{{.Prompt}}}}"
"""
    modelfile_path.write_text(content)
    logger.info(f"Modelfile written to {modelfile_path}")
    return modelfile_path


def register_with_ollama(modelfile_path: Path, model_name: str):
    """Register the model with ollama."""
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        logger.warning("ollama not found in PATH. Install: https://ollama.ai")
        logger.info(f"To register manually: ollama create {model_name} -f {modelfile_path}")
        return

    logger.info(f"Registering with ollama as '{model_name}'...")
    subprocess.run(
        [ollama_bin, "create", model_name, "-f", str(modelfile_path)],
        check=True,
    )
    logger.info(f"Model registered: ollama run {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Export model to GGUF and register with ollama")
    parser.add_argument("--config", help="Path to YAML config (uses model.checkpoint)")
    parser.add_argument("--checkpoint", help="Direct path to checkpoint (overrides config)")
    parser.add_argument("--quantization", default="q4_k_m", help="GGUF quantization type")
    parser.add_argument("--model-name", default="openproof-tactic", help="Name for ollama registration")
    parser.add_argument("--output-dir", default="exports", help="Output directory")
    parser.add_argument("--skip-ollama", action="store_true", help="Skip ollama registration")
    args = parser.parse_args()

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
    elif args.config:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from openproof_ml.utils.config import load_config
        cfg = load_config(args.config)
        checkpoint_dir = Path(cfg["model"]["checkpoint"])
    else:
        parser.error("Either --config or --checkpoint is required")

    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint not found: {checkpoint_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Merge LoRA if needed
    merged_dir = output_dir / "merged"
    model_dir = merge_lora(checkpoint_dir, merged_dir)

    # Step 2: Convert to GGUF
    gguf_path = output_dir / f"{args.model_name}-{args.quantization}.gguf"
    convert_to_gguf(model_dir, gguf_path, args.quantization)

    # Step 3: Create Modelfile
    modelfile = create_ollama_modelfile(gguf_path, args.model_name)

    # Step 4: Register with ollama
    if not args.skip_ollama:
        register_with_ollama(modelfile, args.model_name)

    print(f"\nExport complete:")
    print(f"  GGUF: {gguf_path}")
    print(f"  Modelfile: {modelfile}")
    if not args.skip_ollama:
        print(f"  ollama: ollama run {args.model_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
