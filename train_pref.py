import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

def format_llama3_dpo(example):
    prompt = f"<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 16 Apr 2026\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    chosen = " " + example['chosen'].lstrip() + "<|eot_id|>"
    rejected = " " + example['rejected'].lstrip() + "<|eot_id|>"
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

def main():
    model_id = "/users/PAS3167/bennettgn/ondemand/finalProject/LlamaPostTrainer/sft_merged"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # 4. Configure LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("BennettGN/Qwen-Chains-of-reasoning-synthetic-dpo-dataset-gsm-symbolic", split="train")
    
    print("Formatting dataset for Llama-3 BPE and Instruct tags...")
    dataset = dataset.map(format_llama3_dpo)
    
    training_args = DPOConfig(
        output_dir="./dpo_g8_llama_instruct_result",
        beta=0.1,                                
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,      
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        max_length=1024,            
        num_train_epochs=5,
        logging_steps=10,
        save_steps=100,
        bf16=True, 
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        dataset_num_proc=8,
        report_to="none"
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    trainer.train()
    trainer.save_model("./final_llama3_2_1b_dpo_qwen_instructed")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()