# eval/run_gsm8k_vllm.py
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def load_model_vllm(model_path: str, adapter_path: str | None = None):
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enable_lora=adapter_path is not None,
        max_lora_rank=32,  # 匹配你 LoRA 的 r
    )
    lora_req = None
    if adapter_path is not None:
        lora_req = LoRARequest("adapter", 1, adapter_path)
    return llm, lora_req


def generate_all_vllm(llm, lora_req, prompts: list[str], max_new_tokens=512):
    sampling_params = SamplingParams(
        temperature=0.0,      # greedy
        max_tokens=max_new_tokens,
        stop=["Question:", "\n\nQuestion"],  # base 模型 few-shot 避免接着编下一题
    )
    # 一次性把所有 prompt 扔进去,vLLM 会自动 continuous batching
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_req,
    )
    # 按输入顺序返回
    return [out.outputs[0].text for out in outputs]