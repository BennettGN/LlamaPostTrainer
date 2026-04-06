from datasets import load_dataset

dataset = load_dataset('json', data_files='/users/PAS3167/bennettgn/ondemand/finalProject/LlamaPostTrainer/meta-llama-Llama-3.2-1B-Instruct_synthetic_dpo_dataset_gsm8k.jsonl')
dataset.push_to_hub("BennettGN/Llama-Chains-of-reasoning-synthetic-dpo-dataset-gsm8k")
