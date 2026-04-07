import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer, extract_prompt


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEFAULT_DATASET_NAME = "allenai/olmo-2-0425-1b-preference-mix"
DEFAULT_OUTPUT_DIR = "./llama32_1b_dpo_from_sft_adapter"
DEFAULT_SFT_ADAPTER_PATH = "./llama3.2_1b_sft_local/final_model"
LLAMA3_CHAT_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DPO training with TRL on single or multiple GPUs.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sft-adapter-path", default=DEFAULT_SFT_ADAPTER_PATH)
    parser.add_argument("--no-sft-adapter", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dataset-num-proc", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--precompute-ref-log-probs", action="store_true")
    parser.add_argument("--disable-chat-template-fallback", action="store_true")
    return parser.parse_args()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "-1"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_device() -> int:
    local_rank = get_local_rank()
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    return local_rank


def choose_precision() -> tuple[torch.dtype | None, bool, bool]:
    if not torch.cuda.is_available():
        return torch.float32, False, False
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, False, True
    return torch.float16, True, False


def resolve_base_model_name(default_name: str, adapter_path: str | None) -> str:
    if not adapter_path:
        return default_name

    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")

    with adapter_config_path.open("r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    return adapter_config.get("base_model_name_or_path", default_name)


def ensure_chat_template(tokenizer: Any, fallback_model_name: str, allow_fallback: bool) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    if not allow_fallback:
        return

    try:
        instruct_tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, use_fast=True)
        instruct_template = getattr(instruct_tokenizer, "chat_template", None)
        if instruct_template:
            tokenizer.chat_template = instruct_template
            return
    except Exception as exc:
        if is_main_process():
            print(f"Warning: failed to load chat template from {fallback_model_name}: {exc}")

    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE


def build_quantization_config(args: argparse.Namespace, dtype: torch.dtype | None) -> BitsAndBytesConfig | None:
    if not args.load_in_4bit:
        return None
    if dtype is None:
        dtype = torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name: str, adapter_path: str | None, args: argparse.Namespace) -> Any:
    tokenizer_source = adapter_path or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    ensure_chat_template(
        tokenizer=tokenizer,
        fallback_model_name=DEFAULT_MODEL_NAME,
        allow_fallback=not args.disable_chat_template_fallback,
    )
    return tokenizer


def build_model_load_kwargs(
    args: argparse.Namespace,
    dtype: torch.dtype | None,
    local_rank: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    quantization_config = build_quantization_config(args, dtype)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        if torch.cuda.is_available():
            target_rank = local_rank if local_rank >= 0 else 0
            kwargs["device_map"] = {"": target_rank}
    elif dtype is not None:
        kwargs["torch_dtype"] = dtype
    return kwargs


def load_model_pair(
    args: argparse.Namespace,
    model_name: str,
    dtype: torch.dtype | None,
    local_rank: int,
) -> tuple[Any, Any]:
    load_kwargs = build_model_load_kwargs(args, dtype, local_rank)

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    ref_base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if args.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()
    if args.gradient_checkpointing and hasattr(ref_base_model, "gradient_checkpointing_enable"):
        ref_base_model.gradient_checkpointing_enable()

    if args.sft_adapter_path:
        model = PeftModel.from_pretrained(
            base_model,
            args.sft_adapter_path,
            is_trainable=True,
        )
        ref_model = PeftModel.from_pretrained(
            ref_base_model,
            args.sft_adapter_path,
            is_trainable=False,
        )
    else:
        model = base_model
        ref_model = ref_base_model

    model.config.use_cache = False
    ref_model.config.use_cache = False
    return model, ref_model


def build_peft_config(args: argparse.Namespace) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def load_preference_dataset(args: argparse.Namespace) -> Any:
    train_dataset = load_dataset(args.dataset_name, split="train")

    train_dataset = train_dataset.map(
        extract_prompt,
        desc="Extracting prompt from chosen/rejected",
        num_proc=max(1, args.dataset_num_proc),
    )

    keep_cols = {"prompt", "chosen", "rejected"}
    remove_cols = [c for c in train_dataset.column_names if c not in keep_cols]
    train_dataset = train_dataset.remove_columns(remove_cols)

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), args.max_train_samples)))

    if is_main_process():
        print(train_dataset[0])

    return train_dataset


def build_training_args(args: argparse.Namespace, fp16: bool, bf16: bool) -> DPOConfig:
    training_kwargs: dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_every,
        "save_strategy": "steps",
        "fp16": fp16,
        "bf16": bf16,
        "beta": args.beta,
        "max_length": args.max_length,
        "dataset_num_proc": max(1, args.dataset_num_proc),
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "ddp_find_unused_parameters": False if get_world_size() > 1 else None,
        "precompute_ref_log_probs": args.precompute_ref_log_probs,
    }
    training_kwargs = {key: value for key, value in training_kwargs.items() if value is not None}
    return DPOConfig(**training_kwargs)


def main() -> None:
    args = parse_args()
    if args.no_sft_adapter:
        args.sft_adapter_path = None
    local_rank = setup_device()
    dtype, fp16, bf16 = choose_precision()

    if is_main_process():
        device_name = torch.cuda.get_device_name(local_rank if local_rank >= 0 else 0) if torch.cuda.is_available() else "cpu"
        print(
            f"Starting DPO training | world_size={get_world_size()} | local_rank={local_rank} | "
            f"device={device_name} | fp16={fp16} | bf16={bf16} | load_in_4bit={args.load_in_4bit}"
        )

    model_name = resolve_base_model_name(args.model_name, args.sft_adapter_path)
    tokenizer = load_tokenizer(model_name, args.sft_adapter_path, args)
    model, ref_model = load_model_pair(args, model_name, dtype, local_rank)
    train_dataset = load_preference_dataset(args)
    training_args = build_training_args(args, fp16=fp16, bf16=bf16)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "ref_model": ref_model,
        "args": training_args,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
    }
    if not args.sft_adapter_path:
        trainer_kwargs["peft_config"] = build_peft_config(args)

    trainer = DPOTrainer(**trainer_kwargs)
    trainer.train()

    if is_main_process():
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()

