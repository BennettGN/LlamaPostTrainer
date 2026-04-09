def check_word_count(text: str, min_words: int = None, max_words: int = None) -> bool:
    n = len(text.split())
    if min_words and n < min_words: return False
    if max_words and n > max_words: return False
    return True

def check_contains_keyword(text: str, keyword: str) -> bool:
    return keyword.lower() in text.lower()

def check_json_format(text: str) -> bool:
    import json
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def format_reward(completion: str, constraints: dict) -> float:
    """constraints 是可验证约束的字典"""
    checks = []
    if "min_words" in constraints or "max_words" in constraints:
        checks.append(check_word_count(
            completion, 
            constraints.get("min_words"), 
            constraints.get("max_words")
        ))
    if "keyword" in constraints:
        checks.append(check_contains_keyword(completion, constraints["keyword"]))
    if constraints.get("json_format"):
        checks.append(check_json_format(completion))
    
    if not checks:
        return 0.0
    return sum(checks) / len(checks)  # 平均通过率
