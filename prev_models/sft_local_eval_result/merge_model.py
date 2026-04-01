"""
Merge a PEFT/LoRA adapter into its base model and save a full Hugging Face model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument(
        "--base-model",
        default="meta-llama/Llama-3.2-1B",
        help="Base model name or local path.",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to the PEFT adapter directory.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory to save the merged full model.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype to use while loading for merge.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to from_pretrained. Use 'cpu' to force CPU merge.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> str | torch.dtype:
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    merged_model = model.merge_and_unload()

    merged_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    print(f"Merged model saved to {output_path}")


if __name__ == "__main__":
    main()
