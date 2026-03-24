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
        
        print("Tokenizing datasets...")
        max_length = self.training_args.get("max_seq_length", 1024)
        
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
                tinker_data = []
                batch_size = batch["input_ids"].size(0)
                
                for i in range(batch_size):
                    input_ids_tensor = batch["input_ids"][i]
                    labels_tensor = batch["labels"][i]
                    
                    # 1. Translate HF -100 logic into Tinker's required weights array (1.0 for keep, 0.0 for ignore)
                    weights_tensor = (labels_tensor != -100).float()
                    
                    # 2. Scrub the -100s from the targets so they don't cause out-of-bounds embedding errors on the GPUs
                    target_tokens_tensor = torch.where(labels_tensor == -100, torch.tensor(0), labels_tensor)
                    
                    # 3. Package into Tinker's format using the native PyTorch helpers
                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(input_ids_tensor.tolist()),
                        loss_fn_inputs={
                            "target_tokens": types.TensorData.from_torch(target_tokens_tensor),
                            "weights": types.TensorData.from_torch(weights_tensor)
                        } 
                    )
                    tinker_data.append(datum)
                
                # 4. Send to Tinker's remote GPUs
                fwdbwd_future = self.training_client.forward_backward(
                    data=tinker_data, 
                    loss_fn="cross_entropy"
                )
                
                # 5. Wait for gradients and extract the loss
                fwdbwd_result = fwdbwd_future.result()
                
                # Tinker 2026 SDK uses metrics['loss:sum'] for Cross-Entropy
                if 'loss:sum' in fwdbwd_result.metrics:
                    # We divide by batch size to get the mean loss
                    loss = fwdbwd_result.metrics['loss:sum'] / batch_size
                else:
                    # Fallback if they change the key name
                    loss = sum(fwdbwd_result.metrics.values()) / batch_size 
                
                total_train_loss += loss
                
                # 6. Send the optimizer step with Tinker's AdamParams
                current_lr = scheduler.get_last_lr()[0]
                optim_future = self.training_client.optim_step(
                    types.AdamParams(learning_rate=current_lr)
                )
                optim_future.result() 
                
                scheduler.step()
                
                if step % self.training_args.get("logging_steps", 10) == 0:
                    print(f"Step {step} | Loss: {loss:.4f} | LR: {current_lr:.2e}")
            
            print(f"Epoch {epoch + 1} completed | Avg Train Loss: {total_train_loss/len(train_loader):.4f}")
            
        # 7. Save the checkpoint on Tinker servers
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
    
    # Change to desired hyper param file
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