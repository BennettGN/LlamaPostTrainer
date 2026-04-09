import re

def extract_answer(text: str) -> str | None:
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None

def math_reward(completion: str, ground_truth: str) -> float:
    extracted = extract_answer(completion)
    if extracted is None:
        return 0.0
    try:
        return 1.0 if float(extracted) == float(ground_truth) else 0.0
    except ValueError:
        return 1.0 if extracted.strip() == ground_truth.strip() else 0.0
