from transformers import AutoTokenizer, AutoModel
import torch

# 模型路径，改成你本地路径
model_path = "/Library/MyFolder/Models/embedding/tei/vector_search/BAAI_bge-large-zh-v1.5"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# 嵌入函数
def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # 取 [CLS] 向量
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings[0].numpy()

# 测试调用
if __name__ == "__main__":
    text = "你好，我是 Mac 用户"
    embedding = get_embedding(text)
    print(f"嵌入向量维度: {embedding.shape}")
    print(embedding[:10])  # 打印前10维
