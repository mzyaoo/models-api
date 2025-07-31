import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 选择设备：优先使用 Apple M1/M2 的 GPU（MPS），否则使用 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 本地模型路径
model_path = "/Library/MyFolder/Models/embedding/tei/rerank/BAAI_bge_reranker_large"

# 加载 tokenizer 和模型并移动到对应设备
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # 设置为推理模式，关闭 dropout 等


def rerank(query: str, docs: list[str]) -> list[tuple[str, float]]:
    start_time = time.time()

    """
    对文档列表按照与 query 的匹配程度进行 rerank，返回按 score 降序排序的列表。

    Args:
        query (str): 查询语句
        docs (list[str]): 候选文档列表

    Returns:
        list[tuple[str, float]]: 每个文档及其得分，按得分从高到低排序
    """
    if not docs:
        return []

    # 构建句对：(query, doc)
    pairs = [(query, doc) for doc in docs]

    # 批量编码句对
    inputs = tokenizer.batch_encode_plus(
        pairs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # 将输入数据移动到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits  # 输出 shape: (batch_size, 1)

    # 提取分数
    scores = logits.squeeze(-1).tolist()
    results = list(zip(docs, scores))

    end_time = time.time()
    duration = end_time - start_time
    print(f"生成嵌入耗时：{duration:.4f} 秒")

    # 按分数排序（从高到低）
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
