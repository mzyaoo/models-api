from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def chat():
    model_dir = "/Library/MyFolder/Models/Qwen/Qwen3-4B"

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    prompt = "你是谁？请用一句话介绍你自己。"
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # 可选，是否生成“思考过程”
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("回答：", response)

if __name__ == '__main__':
    chat()