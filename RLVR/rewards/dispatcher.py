from .math_reward import math_reward
from .code_reward import run_unit_tests
from .format_reward import format_reward

def compute_reward(completion: str, sample: dict) -> tuple[float, dict]:
    """
    sample 是数据集的一行,必须有 task_type 字段
    返回 (total_reward, breakdown) 便于分析
    """
    task_type = sample["task_type"]
    breakdown = {}
    
    if task_type == "math":
        r = math_reward(completion, sample["ground_truth"])
        breakdown["correctness"] = r
    elif task_type == "code":
        r = run_unit_tests(completion, sample["test_code"])
        breakdown["unit_test"] = r
    elif task_type == "format":
        r = format_reward(completion, sample["constraints"])
        breakdown["format"] = r
    else:
        print(f"Unknown task type: {task_type}")
        r = 0.0
    
    return r, breakdown
