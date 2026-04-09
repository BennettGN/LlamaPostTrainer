import os

from anyio import Path
import signal
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model
import sys
sys.path.append("/work/experiment/cse5525_project/RLVR")
import torch
from rewards.dispatcher import compute_reward
from data.load_data import load_gsm8k_as_rlvr

def reward_fn(prompts, completions, **kwargs):
    rewards = []
    indexable_kwargs = {k: v for k, v in kwargs.items() if hasattr(v, '__getitem__')}

    for i, (p, c) in enumerate(zip(prompts, completions)):
        sample = {k: v[i] for k, v in indexable_kwargs.items()}
        sample["prompt"] = p
        r, _ = compute_reward(c, sample)
        rewards.append(r)

    return rewards

peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",],
    task_type="CAUSAL_LM",
)

model_path = "/work/experiment/cse5525_project/llama3.2_1b_sft_full"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="bfloat16")
# model = get_peft_model(model, peft_config)
# model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained(model_path)

ref_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="bfloat16")

#dataset = load_mixed_rlvr().select(range(100))  # 先只用 100 条
# dataset = load_gsm8k_as_rlvr().select(range(100))

torch.set_float32_matmul_precision("high")

dataset = load_gsm8k_as_rlvr()

output_dir="./rlvr_1"

args = GRPOConfig(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    num_generations=8,
    learning_rate=5e-6,
    bf16=True,
    logging_steps=1,
    save_steps=100,
    save_total_limit=3,
)

# output_dir = Path("./checkpoints")

# def save_on_exit(signum, frame):
#     print("\nCtrl-C detected! Saving checkpoint...")
#     trainer.save_model(output_dir)  # 保存模型权重
#     if hasattr(trainer, "tokenizer") and trainer.tokenizer:
#         trainer.tokenizer.save_pretrained(output_dir)
#     sys.exit(0)

# signal.signal(signal.SIGINT, save_on_exit)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_fn,
    args=args,
    train_dataset=dataset,
    peft_config=peft_config,
)
trainer.ref_model = ref_model
print(f"Before train: global_step = {trainer.state.global_step if hasattr(trainer, 'state') else 'N/A'}")
trainer.train(resume_from_checkpoint=output_dir + "/checkpoint-934/")
print(f"After train: global_step = {trainer.state.global_step}")

trainer.save_model(os.path.join(output_dir, "final"))
