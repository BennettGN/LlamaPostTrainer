import torch
from datasets import load_dataset,get_dataset_split_names
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

def main():

    model_id = "BennettGN/LLamaSFT3.2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
            "{{ content }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        )

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

    dataset = load_dataset("allenai/olmo-2-0425-1b-preference-mix", split="train")
    print(dataset[0]['chosen'])
    print(dataset[0]['rejected'])
    training_args = DPOConfig(
        output_dir="./dpo_llama3_2_1b_output",
        beta=0.1,                                
        per_device_train_batch_size=64,
        gradient_accumulation_steps=4,   
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        max_length=512,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=100,
        bf16=True, 
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        dataset_num_proc=8,
        report_to="none"
    )

    # 7. Initialize DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 8. Train and Save
    print("Starting DPO training...")
    trainer.train()
    trainer.save_model("./final_llama3_2_1b_dpo")
    print("Training complete and model saved.")

if __name__ == "__main__":
    main()