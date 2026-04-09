import re
import subprocess
import tempfile
import os

def extract_code(completion: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", completion, re.DOTALL)
    return match.group(1) if match else completion

def run_unit_tests(completion: str, test_code: str, timeout: float = 5.0) -> float:
    code = extract_code(completion)
    full_script = f"{code}\n\n{test_code}"
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_script)
        tmp_path = f.name
    
    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    finally:
        os.unlink(tmp_path)
