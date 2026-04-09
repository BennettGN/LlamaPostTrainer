import os
import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
sys.path.append("/work/experiment/cse5525_project/RLVR")
from eval.run_gsm8k_vllm import load_model_vllm, generate_all_vllm
from rewards.math_reward import extract_answer


# ---------- Prompt 构造 ---------- #

FEW_SHOT_EXAMPLES = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "In April, Natalia sold 48 clips. In May, she sold 48 / 2 = 24 clips. Altogether, she sold 48 + 24 = 72 clips.\n#### 72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer": "Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $10.\n#### 10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer": "Betty has 100 / 2 = $50. Her grandparents gave her 15 * 2 = $30. In total she has 50 + 15 + 30 = $95. She needs 100 - 95 = $5 more.\n#### 5"
    },
]

SYSTEM_PROMPT = (
    "You are a helpful math assistant. Solve the problem step by step, "
    "then give the final numerical answer after '####'."
)


def build_prompt_base(question: str, n_shot: int = 8) -> str:
    """Base 模型用 few-shot 续写"""
    shots = FEW_SHOT_EXAMPLES[:n_shot]
    parts = []
    for ex in shots:
        parts.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def build_prompt_chat(question: str, tokenizer, n_shot: int = 0) -> str:
    """Instruct/SFT 模型用 chat template"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # few-shot as prior turns(可选)
    for ex in FEW_SHOT_EXAMPLES[:n_shot]:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------- GSM8K 特有的答案提取 ---------- #

def extract_gt_answer(answer_text: str) -> str:
    """GSM8K ground truth 格式: '推理过程\n#### 数字'"""
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return answer_text.strip()


def extract_pred_answer(completion: str) -> str | None:
    """
    从模型输出里提取答案。
    先尝试 #### 格式,找不到就退而求其次取最后一个数字。
    """
    # 优先: #### N 格式
    ans = extract_answer(completion)  # 复用 math_reward 的函数
    if ans is not None:
        return ans.replace(",", "")
    
    # 退路: 取最后一个数字(base 模型可能不守格式)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", completion)
    if numbers:
        return numbers[-1]
    return None


def is_correct(pred: str | None, gt: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return pred.strip() == gt.strip()


# ---------- 模型加载 ---------- #

def load_model(
    model_path: str,
    adapter_path: str | None = None,
    dtype: str = "bfloat16",
):
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, dtype),
        device_map="auto",
    )
    if adapter_path is not None:
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 生成任务必须 left padding
    
    return model, tokenizer


# ---------- 批量推理 ---------- #

@torch.no_grad()
def generate_batch(
    model, tokenizer, prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> list[str]:
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=2048,
    ).to(model.device)
    
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
    
    outputs = model.generate(**inputs, **gen_kwargs)
    
    # 只保留新生成的部分
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[:, input_len:]
    completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return completions


# ---------- 主评测循环 ---------- #

def evaluate_gsm8k(
    llm, lora_req, tokenizer,
    mode: str = "chat",         # "chat" | "base"
    n_shot: int = 0,             # chat 默认 0-shot,base 默认 8-shot
    batch_size: int = 8,
    max_new_tokens: int = 512,
    max_samples: int | None = None,
    output_file: str | None = None,
) -> dict:
    
    print("Loading GSM8K test split...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    
    # 构造 prompts
    if mode == "chat":
        if n_shot == 0:
            default_shot = 0
        else:
            default_shot = n_shot
        prompts = [
            build_prompt_chat(q, tokenizer, n_shot=default_shot)
            for q in ds["question"]
        ]
    else:  # base
        default_shot = n_shot if n_shot > 0 else 8
        prompts = [
            build_prompt_base(q, n_shot=default_shot)
            for q in ds["question"]
        ]
    
    gts = [extract_gt_answer(a) for a in ds["answer"]]
    questions = list(ds["question"])
    
    # 批量生成
    all_completions = generate_all_vllm(
        llm, lora_req, prompts, max_new_tokens=max_new_tokens
    )
    
    # 判分
    results = []
    n_correct = 0
    n_no_answer = 0
    for q, comp, gt in zip(questions, all_completions, gts):
        pred = extract_pred_answer(comp)
        correct = is_correct(pred, gt)
        if pred is None:
            n_no_answer += 1
        if correct:
            n_correct += 1
        results.append({
            "question": q,
            "completion": comp,
            "prediction": pred,
            "ground_truth": gt,
            "correct": correct,
        })
    
    accuracy = n_correct / len(results)
    no_answer_rate = n_no_answer / len(results)
    
    summary = {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": len(results),
        "n_no_answer": n_no_answer,
        "no_answer_rate": no_answer_rate,
        "mode": mode,
        "n_shot": default_shot,
    }
    
    print("\n=== GSM8K Results ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    
    # 保存详细结果
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({"summary": summary, "details": results}, f, indent=2, ensure_ascii=False)
        print(f"\nDetails saved to {output_file}")
    
    return summary


# ---------- CLI ---------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True,
                   help="Base model 路径,或 merged model 路径")
    p.add_argument("--adapter_path", type=str, default=None,
                   help="可选: LoRA adapter 路径")
    p.add_argument("--mode", choices=["chat", "base"], default="chat",
                   help="chat: 用 chat template; base: few-shot 续写")
    p.add_argument("--n_shot", type=int, default=0,
                   help="few-shot 数量; chat 模式默认 0, base 模式默认 8")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None,
                   help="调试用,只跑前 N 条")
    p.add_argument("--output_file", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    llm, lora_req, tokenizer = load_model_vllm(args.model_path, args.adapter_path)    
    evaluate_gsm8k(
        llm, lora_req, tokenizer,
        mode=args.mode,
        n_shot=args.n_shot,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
