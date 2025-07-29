from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# 本地模型路径
model_path = "/Library/MyFolder/Models/embedding/tei/rerank/BAAI_bge_reranker_large"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def rerank(query: str, docs: list[str]):
    results = []
    for doc in docs:
        # 输入为句对 (query, document)
        inputs = tokenizer(query, doc, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
            # 对于 bge-reranker，logits 是 (1,) 回归值
            score = logits[0].item()
            results.append((doc, score))
    return sorted(results, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    query = "中国的首都是哪里？"
    docs = [
        "中国的首都是北京。",
        "法国的首都是巴黎。",
        "中国有很多城市，比如上海和广州。",
        "我喜欢吃火锅。",
    ]

    reranked = rerank(query, docs)

    print("\nRerank 结果：")
    for rank, (doc, score) in enumerate(reranked, start=1):
        print(f"{rank}. [score={score:.4f}] {doc}")
