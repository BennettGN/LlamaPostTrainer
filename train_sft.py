"""
This module implements the SFTTrainer class for training your model using 
supervised fine-tuning (SFT) via the Tinker API.
"""
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tinker import types
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from trl import DataCollatorForCompletionOnlyLM
import tinker
from dotenv import load_dotenv
load_dotenv()

class TinkerSFTTrainer:
    def __init__(self, tokenizer, training_args):
        self.tokenizer = tokenizer
        self.training_args = training_args
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Authenticating with Tinker...")
        self.service_client = tinker.ServiceClient()
        
        model_id = self.training_args.get("base_model_path", "meta-llama/Llama-3.2-1B")
        print(f"Requesting remote LoRA session for {model_id}...")
        self.training_client = self.service_client.create_lora_training_client(
            base_model=model_id,
            rank=self.training_args.get("lora_rank", 16)
        )

    def _prepare_dataloaders(self):
        dataset_path = self.training_args.get("dataset_path", "allenai/tulu-3-sft-mixture")
        print(f"Downloading dataset: {dataset_path}...")
        load_full = self.training_args.get("load_full_dataset", False)
        
        if load_full:
            print("Loading FULL dataset...")
            full_dataset = load_dataset(dataset_path, split="train") 
        else:
            print("Loading Partial dataset...")
            full_dataset = load_dataset(dataset_path, split="train[:1000]") 
        
        split_ds = full_dataset.train_test_split(test_size=0.05, seed=42)
        train_ds = split_ds["train"]
        val_ds = split_ds["test"]

        local_tokenizer = self.tokenizer 

        def apply_template(example):
            text = local_tokenizer.apply_chat_template(
                example["messages"], 
                tokenize=False, 
                add_generation_prompt=False
            )
            return {"text": text}

        print("Applying chat templates to datasets...")
        train_ds = train_ds.map(apply_template, num_proc=4)
        val_ds = val_ds.map(apply_template, num_proc=4)
        
        max_length = self.training_args.get("max_seq_length", 4096)
        
        # === NEW: Data Filter to prevent Truncation / NaN Loss ===
        print(f"Filtering out sequences longer than {max_length} tokens...")
        train_ds = train_ds.filter(lambda x: len(local_tokenizer.encode(x["text"])) <= max_length)
        val_ds = val_ds.filter(lambda x: len(local_tokenizer.encode(x["text"])) <= max_length)

        print("Tokenizing datasets...")
        def tokenize_function(examples):
            return local_tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=max_length
            )

        train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=train_ds.column_names)
        val_ds = val_ds.map(tokenize_function, batched=True, remove_columns=val_ds.column_names)

        response_template = self.training_args.get("base_model_response_template","<|start_header_id|>assistant<|end_header_id|>\n\n")
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, 
            tokenizer=self.tokenizer,
            mlm=False 
        )

        batch_size = self.training_args.get("batch_size", 1) # Forced to 1 to prevent OOM
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

        return train_loader, val_loader

    def train(self):
        train_loader, val_loader = self._prepare_dataloaders()
        num_epochs = self.training_args.get("num_epochs", 1)
        accum_steps = self.training_args.get("gradient_accumulation_steps", 16)
        
        base_lr = float(self.training_args.get("learning_rate", 2e-5))
        num_training_steps = (len(train_loader) // accum_steps) * num_epochs
        
        dummy_tensor = torch.zeros(1, requires_grad=True)
        dummy_optimizer = torch.optim.AdamW([dummy_tensor], lr=base_lr)
        scheduler = get_cosine_schedule_with_warmup(
            dummy_optimizer,
            num_warmup_steps=self.training_args.get("warmup_steps", 100),
            num_training_steps=num_training_steps
        )

        print("Starting SFT Training via Tinker API...")
        
        # === NEW: Graceful Exit Wrapper ===
        try:
            for epoch in range(num_epochs):
                print(f"--- Starting Epoch {epoch + 1}/{num_epochs} ---")
                total_train_loss = 0
                
                for step, batch in enumerate(train_loader):
                    tinker_data = []
                    batch_size = batch["input_ids"].size(0)
                    
                    for i in range(batch_size):
                        input_ids_tensor = batch["input_ids"][i]
                        labels_tensor = batch["labels"][i]
                        weights_tensor = (labels_tensor != -100).float()
                        target_tokens_tensor = torch.where(labels_tensor == -100, torch.tensor(0), labels_tensor)
                        
                        datum = types.Datum(
                            model_input=types.ModelInput.from_ints(input_ids_tensor.tolist()),
                            loss_fn_inputs={
                                "target_tokens": types.TensorData.from_torch(target_tokens_tensor),
                                "weights": types.TensorData.from_torch(weights_tensor)
                            } 
                        )
                        tinker_data.append(datum)
                    
                    fwdbwd_future = self.training_client.forward_backward(
                        data=tinker_data, 
                        loss_fn="cross_entropy"
                    )
                    fwdbwd_result = fwdbwd_future.result()
                    
                    # 1. Calculate the MEAN loss per token (instead of the massive sum)
                    active_tokens = weights_tensor.sum().item()
                    if active_tokens > 0 and 'loss:sum' in fwdbwd_result.metrics:
                        loss = fwdbwd_result.metrics['loss:sum'] / active_tokens
                    else:
                        loss = sum(fwdbwd_result.metrics.values()) / batch_size 
                    
                    total_train_loss += loss
                    
                    # 2. Gradient Accumulation
                    if (step + 1) % accum_steps == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        optim_future = self.training_client.optim_step(
                            types.AdamParams(learning_rate=current_lr)
                        )
                        optim_future.result() 
                        
                        # Fix the PyTorch Warning!
                        dummy_optimizer.step()
                        scheduler.step()
                        dummy_optimizer.zero_grad()

                    curr_lr = scheduler.get_last_lr()[0] if step >= accum_steps else base_lr
                    print(f"Step {step} | Loss: {loss:.4f} | LR: {curr_lr:.2e}")

                    # === NEW: Intermediate Checkpointing ===
                    if step > 0 and step % 5000 == 0:
                        chkpt_name = f"sft_checkpoint_ep{epoch}_step{step}"
                        print(f"\n[Checkpoint] Saving intermediate weights to {chkpt_name}...")
                        self.training_client.save_weights_for_sampler(name=chkpt_name)
                
                print(f"Epoch {epoch + 1} completed | Avg Train Loss: {total_train_loss/len(train_loader):.4f}")

        except KeyboardInterrupt:
            # === NEW: Rescue logic on Ctrl+C ===
            print("\n\n[WARNING] Training interrupted by user (Ctrl+C)!")
            print("Attempting to rescue and save current model state...")
            rescue_name = f"sft_RESCUE_step{step}"
            self.training_client.save_weights_for_sampler(name=rescue_name)
            print(f"Rescue complete. Saved as: {rescue_name}")
            return 
            
        checkpoint_name = "sft_final_adapter"
        print(f"Compiling and saving LoRA adapter to Tinker cloud as '{checkpoint_name}'...")
        future = self.training_client.save_weights_for_sampler(name=checkpoint_name)
        saved_path = future.result().path
        print(f"SUCCESS! Adapter saved to: {saved_path}")

if __name__ == "__main__":
    if "TINKER_API_KEY" not in os.environ:
        print("WARNING: TINKER_API_KEY environment variable not found.")
        print("Please run: export TINKER_API_KEY='your_key' before executing.")
    
    with open("configs/hyper_params_llama.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    sft_args = config["sft"]
    model_args = config["model"]
    combined_args = {**model_args, **sft_args} 
    
    model_id = combined_args["base_model_path"]
    
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.chat_template = "{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}"

    trainer = TinkerSFTTrainer(
        tokenizer=tokenizer,
        training_args=combined_args
    )
    
    trainer.train()