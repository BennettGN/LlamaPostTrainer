"""
This module implements the PREFTrainer class for training your model using preference optimization.
"""
import os
import yaml
import torch
import torch.nn.functional as F
import asyncio
from tinker import types
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import tinker
from dotenv import load_dotenv

load_dotenv()

def dpo_collate_fn(batch):
    """
    Tells the DataLoader to return a dictionary of lists 
    instead of trying to stack variable-length sequences into tensors.
    """
    return {
        "chosen_input_ids": [item["chosen_input_ids"] for item in batch],
        "chosen_weights": [item["chosen_weights"] for item in batch],
        "chosen_target_tokens": [item["chosen_target_tokens"] for item in batch],
        "rejected_input_ids": [item["rejected_input_ids"] for item in batch],
        "rejected_weights": [item["rejected_weights"] for item in batch],
        "rejected_target_tokens": [item["rejected_target_tokens"] for item in batch],
    }

#Taken from tinker cookbook
def compute_dpo_loss(chosen_logprobs, rejected_logprobs, chosen_ref_logprobs, rejected_ref_logprobs, dpo_beta):
    """
    Computes the scalar DPO loss and basic metrics from paired log probabilities.
    """
    chosen_log_ratio = torch.stack([lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs)])
    rejected_log_ratio = torch.stack([lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs)])

    losses = -F.logsigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio))
    loss = losses.mean()

    accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
    margin = (dpo_beta * chosen_log_ratio - dpo_beta * rejected_log_ratio).mean().item()

    metrics = {"loss:sum": loss.item(), "accuracy": accuracy, "margin": margin}
    return loss, metrics

class PREFTrainer:
    def __init__(self, tokenizer, train_dataset, val_dataset, training_args, checkpoint_model_uri):
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_args = training_args
        self.checkpoint_model_uri = checkpoint_model_uri

        print("Authenticating with Tinker API...")
        self.service_client = tinker.ServiceClient()

        lora_rank = self.training_args.get("lora_rank", 8)

        print(f"Requesting remote DPO LoRA session...")
        print(f"Initializing weights from SFT Checkpoint: {self.checkpoint_model_uri}")

        # Run from previous checkpoint
        self.training_client = self.service_client.create_training_client_from_state(
            self.checkpoint_model_uri, 
        )
        
        print("Forking reference model sampling client...")
        self.reference_client = self.training_client.save_weights_and_get_sampling_client()

    def train(self):
            from torch.utils.data import DataLoader
            
            # 1. Setup Hyperparameters
            # Adjust micro_batch_size based on your GPU memory (e.g., 4 or 8)
            micro_batch_size = self.training_args.get("micro_batch_size", 4)
            num_epochs = self.training_args.get("num_epochs", 1)
            accum_steps = self.training_args.get("gradient_accumulation_steps", 16)
            base_lr = float(self.training_args.get("learning_rate", 5e-7))
            dpo_beta = float(self.training_args.get("beta", 0.1))
            
            # Use a DataLoader to feed chunks of pairs
            train_loader = DataLoader(self.train_dataset, batch_size=micro_batch_size, shuffle=True, collate_fn=dpo_collate_fn)
            num_training_steps = (len(train_loader) // accum_steps) * num_epochs
            
            dummy_tensor = torch.zeros(1, requires_grad=True)
            dummy_optimizer = torch.optim.AdamW([dummy_tensor], lr=base_lr)
            scheduler = get_cosine_schedule_with_warmup(
                dummy_optimizer,
                num_warmup_steps=self.training_args.get("warmup_steps", 100),
                num_training_steps=num_training_steps
            )

            print(f"Starting Batched DPO Training (Micro-batch: {micro_batch_size} pairs)...")
            
            try:
                for epoch in range(num_epochs):
                    print(f"--- Starting Epoch {epoch + 1}/{num_epochs} ---")
                    total_train_loss = 0
                    
                    for step, batch in enumerate(train_loader):
                        # 2. Build the Interleaved Batch: [C0, R0, C1, R1, ...]
                        tinker_data = []
                        current_pairs_in_batch = len(batch["chosen_input_ids"])
                        
                        for i in range(current_pairs_in_batch):
                            # Chosen
                            tinker_data.append(types.Datum(
                                model_input=types.ModelInput.from_ints(batch["chosen_input_ids"][i]),
                                loss_fn_inputs={
                                    "weights": types.TensorData.from_torch(torch.tensor(batch["chosen_weights"][i])),
                                    "target_tokens": types.TensorData.from_torch(torch.tensor(batch["chosen_target_tokens"][i]))
                                }
                            ))
                            # Rejected
                            tinker_data.append(types.Datum(
                                model_input=types.ModelInput.from_ints(batch["rejected_input_ids"][i]),
                                loss_fn_inputs={
                                    "weights": types.TensorData.from_torch(torch.tensor(batch["rejected_weights"][i])),
                                    "target_tokens": types.TensorData.from_torch(torch.tensor(batch["rejected_target_tokens"][i]))
                                }
                            ))

                        # 3. Parallel Reference Computation (Huge speed-up here)
                        async def compute_batch_ref_logprobs():
                            return await asyncio.gather(
                                *[self.reference_client.compute_logprobs_async(d.model_input) for d in tinker_data]
                            )
                        
                        all_ref_logprobs = asyncio.run(compute_batch_ref_logprobs())
                        # Pre-slice ref logprobs [1:] to align with policy logprobs
                        batch_ref_logprob_seqs = [torch.tensor(lp[1:]) for lp in all_ref_logprobs]

                        # 4. Define the Batched Custom Loss closure
                        def dpo_loss_fn(data_list: list[tinker.Datum], policy_logprobs_list: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
                            chosen_logprobs, rejected_logprobs = [], []
                            chosen_ref_logprobs, rejected_ref_logprobs = [], []

                            # Process the batch in pairs
                            for i in range(0, len(data_list), 2):
                                # Active policy logprobs (sliced N-1)
                                p_chosen = policy_logprobs_list[i][1:]
                                p_rejected = policy_logprobs_list[i+1][1:]
                                
                                # Reference logprobs (already sliced N-1)
                                r_chosen = batch_ref_logprob_seqs[i]
                                r_rejected = batch_ref_logprob_seqs[i+1]

                                # Weights (sliced N-1)
                                w_chosen = torch.tensor(data_list[i].loss_fn_inputs["weights"].data, dtype=torch.float32)[1:]
                                w_rejected = torch.tensor(data_list[i+1].loss_fn_inputs["weights"].data, dtype=torch.float32)[1:]

                                # Token Counts for normalization
                                n_c = w_chosen.sum().clamp(min=1.0)
                                n_r = w_rejected.sum().clamp(min=1.0)

                                # Mean Log-probs per token
                                chosen_logprobs.append(torch.dot(p_chosen.float(), w_chosen.float()) / n_c)
                                chosen_ref_logprobs.append(torch.dot(r_chosen.float(), w_chosen.float()) / n_c)
                                
                                rejected_logprobs.append(torch.dot(p_rejected.float(), w_rejected.float()) / n_r)
                                rejected_ref_logprobs.append(torch.dot(r_rejected.float(), w_rejected.float()) / n_r)

                            return compute_dpo_loss(chosen_logprobs, rejected_logprobs, chosen_ref_logprobs, rejected_ref_logprobs, dpo_beta)

                        # 5. Execute Training Step
                        fwdbwd_future = self.training_client.forward_backward_custom(tinker_data, dpo_loss_fn)
                        fwdbwd_result = fwdbwd_future.result()
                        
                        loss = fwdbwd_result.metrics.get('loss:sum', 0.0)
                        total_train_loss += loss
                        
                        # 6. Gradient Accumulation & Scheduler
                        if (step + 1) % accum_steps == 0:
                            current_lr = scheduler.get_last_lr()[0]
                            self.training_client.optim_step(types.AdamParams(learning_rate=current_lr)).result()
                            
                            dummy_optimizer.step()
                            scheduler.step()
                            dummy_optimizer.zero_grad()

                        curr_lr = scheduler.get_last_lr()[0] if step >= accum_steps else base_lr
                        print(f"Step {step} | Loss: {loss:.4f} | Margin: {fwdbwd_result.metrics.get('margin', 0.0):.4f} | Acc: {fwdbwd_result.metrics.get('accuracy', 0.0):.2f} | LR: {curr_lr:.2e}")

                        if step > 0 and step % 5000 == 0:
                            chkpt_name = f"dpo_checkpoint_ep{epoch}_step{step}"
                            self.training_client.save_weights_for_sampler(name=chkpt_name)
                    
                    print(f"Epoch {epoch + 1} completed | Avg Train Loss: {total_train_loss/len(train_loader):.4f}")

            except KeyboardInterrupt:
                print("\n\n[WARNING] Training interrupted!")
                rescue_name = f"dpo_RESCUE_step{step}"
                self.training_client.save_state(name=rescue_name).result()
                return 
                
            print(f"Saving final adapter and state...")
            self.training_client.save_weights_for_sampler(name="dpo_final_adapter").result()
            self.training_client.save_state(name="dpo_final_state").result()
            print(f"Success!")


def format_dpo_dataset(example, tokenizer):
    """
    Ensures all tensors are exactly length N to satisfy Tinker's internal validator.
    """
    if "prompt" in example:
        prompt_text = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
        chosen_text = tokenizer.apply_chat_template(example["prompt"] + example["chosen"], tokenize=False, add_generation_prompt=False)
        rejected_text = tokenizer.apply_chat_template(example["prompt"] + example["rejected"], tokenize=False, add_generation_prompt=False)
    else:
        prompt_messages = example["chosen"][:-1]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        chosen_text = tokenizer.apply_chat_template(example["chosen"], tokenize=False, add_generation_prompt=False)
        rejected_text = tokenizer.apply_chat_template(example["rejected"], tokenize=False, add_generation_prompt=False)

    # 1. Tokenize
    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=1024)["input_ids"]
    prompt_len = len(prompt_ids)

    chosen_ids = tokenizer(chosen_text, truncation=True, max_length=1024)["input_ids"]
    rejected_ids = tokenizer(rejected_text, truncation=True, max_length=1024)["input_ids"]

    # 2. Create Weights
    chosen_weights = [0.0] * prompt_len + [1.0] * (len(chosen_ids) - prompt_len)
    rejected_weights = [0.0] * prompt_len + [1.0] * (len(rejected_ids) - prompt_len)

    # 3. Validation Check: Ensure lengths are consistent
    chosen_weights = chosen_weights[:len(chosen_ids)]
    rejected_weights = rejected_weights[:len(rejected_ids)]

    return {
        "chosen_input_ids": chosen_ids,
        "chosen_weights": chosen_weights,
        "chosen_target_tokens": chosen_ids, # Length N
        "rejected_input_ids": rejected_ids,
        "rejected_weights": rejected_weights,
        "rejected_target_tokens": rejected_ids # Length N
    }

if __name__ == "__main__":
    if "TINKER_API_KEY" not in os.environ:
        print("WARNING: TINKER_API_KEY environment variable not found.")
        print("Please run: export TINKER_API_KEY='your_key' before executing.")
        exit(1)
    
    with open("configs/hyper_params_llama.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    dpo_args = config["dpo"]
    model_args = config["model"]
    combined_args = {**model_args, **dpo_args} 
    
    model_id = combined_args["base_model_path"]
    sft_checkpoint_uri = config.get("latest_model_checkpoint_uri")
    
    if not sft_checkpoint_uri:
        print("ERROR: latest_model_checkpoint_uri is missing from yaml!")
        exit(1)
    
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Lama chat template
    tokenizer.chat_template = "{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = combined_args.get("dataset_path", "allenai/olmo-2-0425-1b-preference-mix")
    print(f"Downloading preference dataset: {dataset_path}...")
    
    load_full = combined_args.get("load_full_dataset", False)
    split_str = "train" if load_full else "train[:1000]"
    full_dataset = load_dataset(dataset_path, split=split_str) 
    
    print("Tokenizing and applying chat templates and weight masks...")
    tokenized_dataset = full_dataset.map(
        lambda x: format_dpo_dataset(x, tokenizer),
        batched=False,
        remove_columns=full_dataset.column_names 
    )

    print("Splitting into train and validation sets...")
    split_ds = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_ds["train"]
    val_dataset = split_ds["test"]

    print(f"Initializing PREFTrainer with SFT base checkpoint: {sft_checkpoint_uri}")
    trainer = PREFTrainer(
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=combined_args,
        checkpoint_model_uri=sft_checkpoint_uri
    )
    
    trainer.train()