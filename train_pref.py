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
        num_epochs = self.training_args.get("num_epochs", 1)
        accum_steps = self.training_args.get("gradient_accumulation_steps", 16)
        base_lr = float(self.training_args.get("learning_rate", 2e-5))
        dpo_beta = float(self.training_args.get("beta", 0.1))
        
        num_training_steps = (len(self.train_dataset) // accum_steps) * num_epochs
        
        # Dummy optimizer trick to utilize huggingface's built-in LR scheduler
        dummy_tensor = torch.zeros(1, requires_grad=True)
        dummy_optimizer = torch.optim.AdamW([dummy_tensor], lr=base_lr)
        scheduler = get_cosine_schedule_with_warmup(
            dummy_optimizer,
            num_warmup_steps=self.training_args.get("warmup_steps", 100),
            num_training_steps=num_training_steps
        )

        print("Starting DPO Training via Tinker API...")
        
        try:
            for epoch in range(num_epochs):
                print(f"--- Starting Epoch {epoch + 1}/{num_epochs} ---")
                total_train_loss = 0
                
                for step, batch in enumerate(self.train_dataset):
                    # 1. Prepare Interleaved Batch Data
                    tinker_data = [
                        types.Datum(
                            model_input=types.ModelInput.from_ints(batch["chosen_input_ids"]),
                            loss_fn_inputs={
                                "weights": types.TensorData.from_torch(torch.tensor(batch["chosen_weights"])),
                                "target_tokens": types.TensorData.from_torch(torch.tensor(batch["chosen_target_tokens"]))
                            }
                        ),
                        types.Datum(
                            model_input=types.ModelInput.from_ints(batch["rejected_input_ids"]),
                            loss_fn_inputs={
                                "weights": types.TensorData.from_torch(torch.tensor(batch["rejected_weights"])),
                                "target_tokens": types.TensorData.from_torch(torch.tensor(batch["rejected_target_tokens"]))
                            }
                        )
                    ]

                    # 2. Asynchronously compute reference logprobs
                    async def compute_all_ref_logprobs():
                        return await asyncio.gather(
                            *[self.reference_client.compute_logprobs_async(datum.model_input) for datum in tinker_data]
                        )
                    
                    all_ref_logprobs = asyncio.run(compute_all_ref_logprobs())
                    
                    # Skip the first token (prompt) since it doesn't have a predicted logprob
                    all_ref_logprob_seqs = [torch.tensor(logprobs[1:]) for logprobs in all_ref_logprobs]
                    chosen_ref_logprob_seqs = [all_ref_logprob_seqs[0]]
                    rejected_ref_logprob_seqs = [all_ref_logprob_seqs[1]]

                    def dpo_loss_fn(data: list[tinker.Datum], logprobs_list: list[torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
                        # 1. Extract logprobs
                        policy_chosen = logprobs_list[0][1:]
                        policy_rejected = logprobs_list[1][1:]
                        ref_chosen = chosen_ref_logprob_seqs[0]
                        ref_rejected = rejected_ref_logprob_seqs[0]

                        # 2. Extract weights
                        weights_chosen = torch.tensor(data[0].loss_fn_inputs["weights"].data, dtype=torch.float32)[1:]
                        weights_rejected = torch.tensor(data[1].loss_fn_inputs["weights"].data, dtype=torch.float32)[1:]

                        # 3. Calculate Token Counts (to avoid division by zero)
                        n_chosen = weights_chosen.sum().clamp(min=1.0)
                        n_rejected = weights_rejected.sum().clamp(min=1.0)

                        # 4. Normalize Logprobs (avoiding cumulative calculation)
                        chosen_logprobs = [(torch.dot(policy_chosen.float(), weights_chosen.float()) / n_chosen)]
                        chosen_ref_logprobs = [(torch.dot(ref_chosen.float(), weights_chosen.float()) / n_chosen)]

                        rejected_logprobs = [(torch.dot(policy_rejected.float(), weights_rejected.float()) / n_rejected)]
                        rejected_ref_logprobs = [(torch.dot(ref_rejected.float(), weights_rejected.float()) / n_rejected)]

                        return compute_dpo_loss(
                            chosen_logprobs, 
                            rejected_logprobs, 
                            chosen_ref_logprobs, 
                            rejected_ref_logprobs, 
                            dpo_beta
                        )

                    fwdbwd_future = self.training_client.forward_backward_custom(tinker_data, dpo_loss_fn)
                    fwdbwd_result = fwdbwd_future.result()
                    
                    # Track Metrics & Accumulate Gradients
                    loss = fwdbwd_result.metrics.get('loss:sum', 0.0)
                    total_train_loss += loss
                    
                    if (step + 1) % accum_steps == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        optim_future = self.training_client.optim_step(
                            types.AdamParams(learning_rate=current_lr)
                        )
                        optim_future.result() 
                        
                        dummy_optimizer.step()
                        scheduler.step()
                        dummy_optimizer.zero_grad()

                    curr_lr = scheduler.get_last_lr()[0] if step >= accum_steps else base_lr
                    print(f"Step {step} | Loss: {loss:.4f} | Margin: {fwdbwd_result.metrics.get('margin', 0.0):.4f} | LR: {curr_lr:.2e}")

                    if step > 0 and step % 5000 == 0:
                        chkpt_name = f"dpo_checkpoint_ep{epoch}_step{step}"
                        print(f"\n[Checkpoint] Saving intermediate weights to {chkpt_name}...")
                        self.training_client.save_weights_for_sampler(name=chkpt_name)
                
                print(f"Epoch {epoch + 1} completed | Avg Train Loss: {total_train_loss/len(self.train_dataset):.4f}")

        except KeyboardInterrupt:
            print("\n\n[WARNING] Training interrupted by user (Ctrl+C)!")
            print("Attempting to rescue and save current model state...")
            rescue_name = f"dpo_RESCUE_step{step}"
            future = self.training_client.save_state(name=rescue_name)
            saved_path = future.result().path
            print(f"Rescue complete. Saved as: {rescue_name}")
            return 
            
        checkpoint_name = "dpo_final_adapter"
        print(f"Compiling and saving LoRA adapter to Tinker cloud as '{checkpoint_name}'...")
        future = self.training_client.save_state(name=checkpoint_name)
        saved_path = future.result().path
        print(f"SUCCESS! Adapter saved to: {saved_path}")


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