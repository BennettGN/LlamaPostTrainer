REQUIRED_FIELDS = ["prompt", "task_type"]

def validate_sample(sample: dict) -> bool:
    if not all(f in sample for f in REQUIRED_FIELDS):
        return False
    t = sample["task_type"]
    if t == "math" and "ground_truth" not in sample:
        return False
    if t == "code" and "test_code" not in sample:
        return False
    if t == "format" and "constraints" not in sample:
        return False
    return True
