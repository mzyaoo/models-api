from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

if __name__ == '__main__':
    model_name = "Qwen/Qwen3-4B"

    # 1. 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # 2. 设置 prompt
    prompt = "你好，你是谁，请使用中文回到我的问题！"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False   # 关闭思考模式，加快生成
    )

    # 3. 准备输入
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 4. 控制生成数量，最多生成 256 个 tokens（非常够用了）
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    print("输出内容:\n", output_text)