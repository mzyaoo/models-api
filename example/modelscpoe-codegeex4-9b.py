from modelscope import AutoTokenizer, AutoModelForCausalLM
from sympy.printing.pytorch import torch

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("ZhipuAI/codegeex4-all-9b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "ZhipuAI/codegeex4-all-9b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": "write a quick sort"}],
                                           add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                           return_dict=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))