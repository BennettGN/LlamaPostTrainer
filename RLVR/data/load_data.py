from datasets import load_dataset, concatenate_datasets

def load_gsm8k_as_rlvr():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    def transform(x):
        return {
            "prompt": x["question"],
            "task_type": "math",
            "ground_truth": x["answer"].split("####")[-1].strip(),
        }
    return ds.map(transform, remove_columns=ds.column_names)

def load_mixed_rlvr():
    math_ds = load_gsm8k_as_rlvr()
    # code_ds = load_code_as_rlvr()
    # format_ds = load_ifeval_style_as_rlvr()
    return math_ds  # 先只用数学,跑通再加
