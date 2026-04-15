"""
Merge a PEFT/LoRA adapter into its base model and save a full Hugging Face model.
Base model is auto-detected from adapter_config.json if not explicitly provided.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name or local path. Auto-detected from adapter_config.json if omitted.",
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


def resolve_base_model(adapter_path: Path, override: str | None) -> str:
    if override:
        return override
    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_path}. "
            "Please specify --base-model explicitly."
        )
    config = json.loads(config_file.read_text())
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            "adapter_config.json does not contain 'base_model_name_or_path'. "
            "Please specify --base-model explicitly."
        )
    print(f"Auto-detected base model: {base_model}")
    return base_model


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    base_model = resolve_base_model(adapter_path, args.base_model)

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
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