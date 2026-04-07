import datasets
from pathlib import Path
import inspect
import time

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    Trainer,
    TrainingArguments as HFTrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model


base_model_name_ = "meta-llama/Llama-3.2-1B-Instruct"
dataset_name_ = "allenai/tulu-3-sft-olmo-2-mixture"
max_length_ = 2048
batch_size_ = 2
num_epochs_ = 1
learning_rate_ = 2e-4
adam_beta1_ = 0.9
adam_beta2_ = 0.95
adam_epsilon_ = 1e-8
weight_decay = 0.1
warmup_steps = 100
gradient_accumulation_steps_ = 4
use_bf16 = True
use_fp16 = False
seed_ = 42
dataloader_num_workers_ = 4
lora_rank_ = 16
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype_ = torch.bfloat16
output_dir = Path("./outputs/llama3.2_1b_sft_local_2")
collator_call_count = 0
sample_every_steps_ = 100
sample_num_return_sequences_ = 5
sample_max_new_tokens_ = 64
sample_prompt_prefix_ = "I'm just a language model"


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = """
{% for message in messages %}
{% if loop.first %}{{ bos_token }}{% endif %}
{% if message['role'] == 'assistant' %}
{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{% generation %}
{{ message['content'] | trim }}
{% endgeneration %}
{{ '<|eot_id|>' }}
{% else %}
{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + (message['content'] | trim) + '<|eot_id|>' }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{% endif %}
"""
    return tokenizer

def prepare_dataset(dataset_name, tokenizer, batch_size):
    dataset = datasets.load_dataset(dataset_name, split="train")

    def apply_template(example):
        encoded = tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=True, 
            return_tensors=None,
            return_dict=True,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False
        )
        input_ids = encoded["input_ids"][:max_length_]
        attn_mask = encoded["attention_mask"][:max_length_]
        assistant_mask = encoded["assistant_masks"][:max_length_]
        labels = [token if assistant_mask[i] == 1 else -100 for i, token in enumerate(input_ids)]
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels, "input_length": len(input_ids)}
    
    train_ds = dataset.map(apply_template, num_proc=10, remove_columns=dataset.column_names)
    return train_ds

def collate_without_aux_columns(features):
    global collator_call_count
    collator_call_count += 1
    if collator_call_count <= 3:
        print(
            f"[collate] building batch #{collator_call_count} with {len(features)} samples",
            flush=True,
        )
    cleaned = []
    for feature in features:
        row = dict(feature)
        row.pop("input_length", None)
        cleaned.append(row)
    return collator(cleaned)


def save_training_artifacts(trainer, tokenizer, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    trainer_state = getattr(trainer, "state", None)
    if trainer_state is not None and hasattr(trainer_state, "save_to_json"):
        trainer_state.save_to_json(str(save_dir / "trainer_state.json"))
    print(f"Saved model artifacts to {save_dir}")


class PeriodicSamplingCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        prompt_prefix: str,
        every_steps: int,
        num_return_sequences: int,
        max_new_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt_prefix = prompt_prefix
        self.every_steps = every_steps
        self.num_return_sequences = num_return_sequences
        self.max_new_tokens = max_new_tokens
        self.last_sampled_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        if state.global_step <= 0 or state.global_step == self.last_sampled_step:
            return control
        if state.global_step % self.every_steps != 0:
            return control

        model = kwargs["model"]
        was_training = model.training
        model.eval()

        prompt_inputs = self.tokenizer(
            self.prompt_prefix,
            return_tensors="pt",
            add_special_tokens=True,
        )
        prompt_inputs = {k: v.to(model.device) for k, v in prompt_inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **prompt_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=self.num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = prompt_inputs["input_ids"].shape[-1]
        print(
            f"\n[sample] step={state.global_step} prompt_prefix={self.prompt_prefix!r}",
            flush=True,
        )
        for idx, sequence in enumerate(outputs, start=1):
            completion = self.tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True)
            print(f"[sample] {idx}: {completion}", flush=True)

        self.last_sampled_step = state.global_step
        if was_training:
            model.train()
        return control


def main() -> None:
    global collator, collator_call_count

    output_dir.mkdir(parents=True, exist_ok=True)

    print("[init] loading tokenizer", flush=True)
    tokenizer = load_tokenizer()
    print("[init] preparing dataset", flush=True)
    train_ds = prepare_dataset(dataset_name_, tokenizer=tokenizer, batch_size=batch_size_)
    print(f"[init] dataset ready with {len(train_ds)} rows", flush=True)

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": num_epochs_,
        "per_device_train_batch_size": batch_size_,
        "per_device_eval_batch_size": batch_size_,
        "gradient_accumulation_steps": gradient_accumulation_steps_,
        "learning_rate": learning_rate_,
        "adam_beta1": adam_beta1_,
        "adam_beta2": adam_beta2_,
        "adam_epsilon": adam_epsilon_,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "eval_strategy": "no",
        "save_strategy": "no",
        "bf16": True,
        "remove_unused_columns": False,
        "seed": seed_,
        "dataloader_pin_memory": True,
        "dataloader_num_workers": dataloader_num_workers_,
        "do_train": True,
        "length_column_name": "input_length",
    }

    training_signature = inspect.signature(HFTrainingArguments.__init__)
    if "train_sampling_strategy" in training_signature.parameters:
        training_kwargs["train_sampling_strategy"] = "group_by_length"
    elif "group_by_length" in training_signature.parameters:
        training_kwargs["group_by_length"] = True

    hf_args = HFTrainingArguments(**training_kwargs)
    print("[init] training arguments ready", flush=True)

    print("[init] loading base model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name_)
    print("[init] attaching LoRA adapters", flush=True)
    lora_config = LoraConfig(
        r=lora_rank_,
        lora_alpha=lora_rank_ * 2,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
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
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    print("[init] model moved to device", flush=True)
    # model = torch.compile(model, mode="reduce-overhead")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        return_tensors="pt",
    )

    trainer_kwargs = {
        "model": model,
        "args": hf_args,
        "train_dataset": train_ds,
        "data_collator": collate_without_aux_columns,
        "processing_class": tokenizer,
        "callbacks": [
            PeriodicSamplingCallback(
                tokenizer=tokenizer,
                prompt_prefix=sample_prompt_prefix_,
                every_steps=sample_every_steps_,
                num_return_sequences=sample_num_return_sequences_,
                max_new_tokens=sample_max_new_tokens_,
            )
        ],
    }

    trainer = Trainer(**trainer_kwargs)
    collator_call_count = 0
    print("[train] trainer initialized", flush=True)
    print("[train] entering trainer.train()", flush=True)
    train_start = time.time()

    try:
        trainer.train()
    except KeyboardInterrupt:
        step = int(getattr(trainer.state, "global_step", 0))
        checkpoint_dir = output_dir / f"interrupt-checkpoint-step-{step:06d}"
        print(f"Training interrupted by Ctrl-C. Saving checkpoint to {checkpoint_dir}")
        save_training_artifacts(trainer, tokenizer, checkpoint_dir)
        raise SystemExit(130)
    finally:
        print(f"[train] trainer.train() returned after {time.time() - train_start:.1f}s", flush=True)

    final_model_dir = output_dir / "final_model"
    save_training_artifacts(trainer, tokenizer, final_model_dir)


if __name__ == "__main__":
    collator = None
    main()

