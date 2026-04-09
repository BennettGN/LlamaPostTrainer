from data.load_data import load_gsm8k_as_rlvr
from rewards.dispatcher import compute_reward

ds = load_gsm8k_as_rlvr()
sample = ds[0]
print("Prompt:", sample["prompt"])
print("GT:", sample["ground_truth"])

# 模拟一个假的 completion
fake_completion = f"Let me think... #### {sample['ground_truth']}"
r, breakdown = compute_reward(fake_completion, sample)
print(f"Reward: {r}, Breakdown: {breakdown}")
assert r == 1.0, "Reward function broken!"
