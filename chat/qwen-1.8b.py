from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == '__main__':
    # ✅ 本地模型路径（你自己替换为你的本地目录）
    model_dir = "/Library/MyFolder/Models/Qwen/Qwen-1_8B-Chat"

    # 加载 tokenizer 和 model（信任远程代码是必须的，因为 Qwen 使用了自定义的模型定义）
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    # 第一轮对话
    response, history = model.chat(tokenizer, "你好", history=None)
    print(response)

    # 第二轮对话
    response, history = model.chat(tokenizer, "给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
    print(response)

    # 第三轮对话
    response, history = model.chat(tokenizer, "给这个故事起一个标题", history=history)
    print(response)

    # 使用 system prompt（设定语气）
    response, _ = model.chat(tokenizer, "你好呀", history=None, system="请用二次元可爱语气和我说话")
    print(response)