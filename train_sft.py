"""
This module implements the SFTTrainer class for training your model using 
supervised fine-tuning (SFT) via the Tinker API.
"""
import os
import yaml
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from trl import DataCollatorForCompletionOnlyLM
import tinker

class TinkerSFTTrainer:
    def __init__(self, tokenizer, training_args):
        self.tokenizer = tokenizer
        self.training_args = training_args
        
        # Ensure tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 1. Initialize the Tinker Client
        print("Authenticating with Tinker...")
        self.service_client = tinker.ServiceClient()
        
        # 2. Spin of LoRa session for Llama
        model_id = self.training_args.get("base_model_path", "meta-llama/Llama-3.2-1B")
        print(f"Requesting remote LoRA session for {model_id}...")
        self.training_client = self.service_client.create_lora_training_client(
            base_model=model_id,
            rank=self.training_args.get("lora_rank", 16)
        )

    def _prepare_dataloaders(self):
        """
        Loads the dataset locally, applies the Llama-3 chat template, 
        and uses TRL to mask the user prompts.
        train_loader: training set
        val_loader: 5% slice of dataset for testing
        """
        dataset_path = self.training_args.get("dataset_path", "allenai/tulu-3-sft-mixture")
        print(f"Downloading dataset: {dataset_path}...")
        load_full = self.training_args.get("load_full_dataset", False)
        
        if load_full:
            print("Loading FULL dataset...")
            full_dataset = load_dataset(dataset_path,split="train") 
        else:
            print("Loading Partial dataset...")
            full_dataset = load_dataset(dataset_path,split="train[:1000]") 
        
        split_ds = full_dataset.train_test_split(test_size=0.05, seed=42)
        train_ds = split_ds["train"]
        val_ds = split_ds["test"]

        def apply_template(example):
            text = self.tokenizer.apply_chat_template(
                example["messages"], 
                tokenize=False, 
                add_generation_prompt=False
            )
            return {"text": text}

        print("Applying chat templates to datasets...")
        train_ds = train_ds.map(apply_template, num_proc=4)
        val_ds = val_ds.map(apply_template, num_proc=4)

        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template, 
            tokenizer=self.tokenizer,
            mlm=False 
        )

        batch_size = self.training_args.get("batch_size", 4)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

        return train_loader, val_loader

    def train(self):
        """
        The main training loop communicating with Tinker.
        """
        train_loader, val_loader = self._prepare_dataloaders()
        num_epochs = self.training_args.get("num_epochs", 3)
        
        base_lr = float(self.training_args.get("learning_rate", 2e-5))
        num_training_steps = len(train_loader) * num_epochs
        
        # Create a dummy optimizer to hook up the HuggingFace scheduler
        dummy_tensor = torch.zeros(1, requires_grad=True)
        dummy_optimizer = torch.optim.AdamW([dummy_tensor], lr=base_lr)
        scheduler = get_cosine_schedule_with_warmup(
            dummy_optimizer,
            num_warmup_steps=self.training_args.get("warmup_steps", 100),
            num_training_steps=num_training_steps
        )

        print("Starting SFT Training via Tinker API...")
        for epoch in range(num_epochs):
            print(f"--- Starting Epoch {epoch + 1}/{num_epochs} ---")
            total_train_loss = 0
            
            for step, batch in enumerate(train_loader):
                # 1. Extract lists to send over the network
                input_ids = batch["input_ids"].tolist()
                labels = batch["labels"].tolist()
                
                # 2. Send batch to Tinker's remote GPUs
                future = self.training_client.forward_backward(
                    input_ids=input_ids,
                    labels=labels
                )
                
                # 3. Wait for the server to process and return the loss
                loss = future.result()
                total_train_loss += loss
                
                current_lr = scheduler.get_last_lr()[0]
                self.training_client.optim_step(lr=current_lr)
                scheduler.step()
                
                if step % self.training_args.get("logging_steps", 10) == 0:
                    print(f"Step {step} | Loss: {loss:.4f} | LR: {current_lr:.2e}")
            
            print(f"Epoch {epoch + 1} completed | Avg Train Loss: {total_train_loss/len(train_loader):.4f}")
            
        # 5. Save the checkpoint on Tinker servers + download the weights
        output_dir = self.training_args.get("output_dir", "./sft_output")
        print(f"Saving model adapter to {output_dir}...")
        
        checkpoint_name = "sft_final"
        print(f"Saving model to Tinker cloud as '{checkpoint_name}'...")
        future = self.training_client.save_state(name=checkpoint_name)
        saved_path = future.result().path
        
        print(f"Successfully saved to: {saved_path}")

if __name__ == "__main__":

    if "TINKER_API_KEY" not in os.environ:
        print("WARNING: TINKER_API_KEY environment variable not found.")
        print("Please run: export TINKER_API_KEY='your_key' before executing.")
        
    with open("configs/hyper_params.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    sft_args = config["sft"]
    model_id = config["model"]["base_model_path"]
    
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    trainer = TinkerSFTTrainer(
        tokenizer=tokenizer,
        training_args=sft_args
    )
    
    trainer.train()