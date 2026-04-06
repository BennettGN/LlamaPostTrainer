import re
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def formatDataset():
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    messages, expectedResponses = [], []
    for row in dataset:
        messages.append({"role": "user", "content": row['question'] + "\n\nGive the final answer on a new line with a prefix of ####"})
        expectedResponses.append(row['answer'])
    return messages, expectedResponses

def extract_dataset_answer(dataset_answer_string):
    parts = dataset_answer_string.split("####")
    if len(parts) > 1:
        return parts[1].strip().replace(",", "").replace("$", "")
    return None

def extract_model_answer(model_response):
    matches = re.findall(r'####\s*(?:[^\d\n]*)?(-?[\d.,]+)', model_response)
    if matches:
        return matches[-1].replace(',', '').rstrip('.')
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', model_response)
    if numbers:
        return numbers[-1].replace(',', '').rstrip('.')
    return None

if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = LLM(model=model_id, dtype="float16")
    
    chains_of_reasoning = 3 
    
    sampling_params = SamplingParams(
        temperature=0.6, 
        max_tokens=512, 
        n=chains_of_reasoning 
    )
    
    messages, expected = formatDataset()
    
    print("Formatting prompts...")
    formatted_prompts = [
        tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True) 
        for msg in messages
    ]

    print("Starting generation...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    
    generated_dataset = []
    safe_model_id = model_id.replace("/", "-")
    file_name = f"{safe_model_id}_synthetic_dpo_dataset_gsm8k.jsonl"

    # 5. Process the outputs
    for i, output in enumerate(outputs):
        original_message = messages[i]
        expected_answer = expected[i]
        final_expected_answer = extract_dataset_answer(expected_answer)
        
        found_correct = False
        found_incorrect = False
        correct_seq = ""
        incorrect_seq = ""

        # output.outputs contains the 'n' generated sequences
        for generated_sequence in output.outputs:
            response_text = generated_sequence.text
            final_actual_answer = extract_model_answer(response_text)
            
            try:
              is_correct = (float(final_actual_answer) == float(final_expected_answer))
            except (ValueError, TypeError):
              is_correct = False
        
            if is_correct and not found_correct:
                correct_seq = response_text.strip()
                found_correct = True
            elif not is_correct and not found_incorrect:
                incorrect_seq = response_text.strip()
                found_incorrect = True

        if found_incorrect:
            chosen_text = correct_seq if correct_seq else expected_answer.strip()
            generated_dataset.append({
                "prompt": original_message["content"], 
                "chosen": chosen_text, 
                "rejected": incorrect_seq
            })

    with open(file_name, "w") as f:
        for response_pair in generated_dataset:
            f.write(json.dumps(response_pair) + "\n") 
    print(f"Data successfully written to {file_name}")