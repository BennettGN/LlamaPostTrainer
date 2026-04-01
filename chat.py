import gradio as gr
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

model_path = "BennettGN/LLamaSFT3.2"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    device_map="auto"
)

print(f"Model device: {next(model.parameters()).device}", flush=True)

if tokenizer.chat_template is None:
    tokenizer.chat_template = (
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    )

terminators = [
    tokenizer.eos_token_id,
    128009
]

def extract_text(content):
    """Handle both string content and Gradio's nested content format."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return " ".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)

def respond(message, history):
    formatted_history = []

    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            content = extract_text(item.get("content", ""))
            if role in ("user", "assistant") and content:
                formatted_history.append({"role": role, "content": content})
        else:
            # fallback for old tuple format
            user_msg, bot_msg = item[0], item[1]
            formatted_history.append({"role": "user", "content": user_msg})
            if bot_msg:
                formatted_history.append({"role": "assistant", "content": bot_msg})

    formatted_history.append({"role": "user", "content": message})

    tokenized = tokenizer.apply_chat_template(
        formatted_history,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    )
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized["attention_mask"].to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        inputs=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.3,
        streamer=streamer
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message

    thread.join()

demo = gr.ChatInterface(
    respond,
    title="My Fine-Tuned Llama Model Eval",
    description="Testing the outputs interactively with real-time streaming."
)

if __name__ == "__main__":
    demo.launch(share=True)