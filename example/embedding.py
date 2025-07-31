from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import time  # 用于计算耗时

# 优先使用 MPS（Apple GPU），否则使用 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 本地模型路径
model_path = "/Library/MyFolder/Models/embedding/tei/vector_search/BAAI_bge-large-zh-v1.5"

# 加载 tokenizer 和模型，并迁移到设备
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()  # 设置为推理模式


# 获取单个文本的嵌入向量，并统计耗时
def get_embedding(text: str) -> np.ndarray:
    start_time = time.time()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] 向量

    embedding_np = embedding[0].cpu().numpy()
    end_time = time.time()

    duration = end_time - start_time
    print(f"生成嵌入耗时：{duration:.4f} 秒")

    return embedding_np


# 测试调用
if __name__ == "__main__":
    text = "你好，我是 Mac 用户"
    embedding = get_embedding(text)

    print(f"嵌入向量维度: {embedding.shape}")
    print("前10维：", embedding[:10])