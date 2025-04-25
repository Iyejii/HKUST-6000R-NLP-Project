import json
import re
from collections import Counter
from typing import List
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

# -----加载 PubMedQA 数据 -----  
def load_pubmedqa_data(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)
    contexts, questions, answers = [], [], []
    for article_id, article_data in data.items():
        question = article_data["QUESTION"]
        context = " ".join(article_data["CONTEXTS"])
        long_answer = article_data["LONG_ANSWER"]
        
        contexts.append(context)
        questions.append(question)
        answers.append(long_answer)
    return contexts, questions, answers

# -----BM25 Retriever -----  
class BM25Retriever:
    def __init__(self, corpus: List[str]):
        self.tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.corpus[i] for i in top_indices]

# ----- Dense Retriever -----  
class DenseRetriever:
    def __init__(self, corpus: list, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.corpus = corpus
        self.embeddings = self.model.encode(corpus, convert_to_tensor=True)

    def retrieve(self, query: str, top_k=3) -> list:
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)[0]
        return [self.corpus[hit['corpus_id']] for hit in hits]

# ----- Hybrid Retriever (BM25 + Dense) -----  
class HybridRetriever:
    def __init__(self, corpus: list, bm25_weight=0.5, dense_weight=0.5, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.bm25 = BM25Retriever(corpus)
        self.dense = DenseRetriever(corpus, model_name)
        self.corpus = corpus
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def retrieve(self, query: str, top_k=3) -> list:
        # BM25 分数
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.bm25.get_scores(tokenized_query)

        # Dense 分数
        query_embedding = self.dense.model.encode(query, convert_to_tensor=True)
        dense_scores = util.cos_sim(query_embedding, self.dense.embeddings)[0].cpu().tolist()

        # 加权融合
        hybrid_scores = [
            self.bm25_weight * bm25_scores[i] + self.dense_weight * dense_scores[i]
            for i in range(len(self.corpus))
        ]

        # 取 top_k
        top_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)[:top_k]
        return [self.corpus[i] for i in top_indices]

# ----- HF Reader -----  
class HFReader:
    def __init__(self, model_name="deepset/bert-base-cased-squad2"):
        self.qa_pipeline = pipeline("question-answering", model=model_name)

    def answer(self, question: str, context: str) -> str:
        result = self.qa_pipeline(question=question, context=context)
        return result["answer"]

# ----- Evaluation Metrics -----  
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return re.sub(r'[^\w\s]', '', text)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

# ----- Main Execution -----  
def run_pipeline(retriever_type="dense", reader_type="hf"):
    # 加载 PubMedQA 数据集
    pubmedqa_file_path = r'C:path\ori_pqau.json'  # 替换PubMedQA 数据文件路径
    corpus, questions, answers = load_pubmedqa_data(pubmedqa_file_path)

    # 只保留前 1000 个问题（如果总数够）
    max_samples = 1000
    corpus = corpus[:max_samples]
    questions = questions[:max_samples]
    answers = answers[:max_samples]

    # 选择检索器
    if retriever_type == "dense":
        retriever = DenseRetriever(corpus)
    elif retriever_type == "bm25":
        retriever = BM25Retriever(corpus)
    elif retriever_type == "hybrid":
        retriever = HybridRetriever(corpus)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    # 选择阅读器
    reader = HFReader() if reader_type == "hf" else None  # 如果需要OpenAI，可以在此处增加处理

    # 初始化一个列表来保存结果
    results = []
    
    total_f1 = 0
    for i, q in enumerate(questions):
        context = " ".join(retriever.retrieve(q, top_k=3))
        pred = reader.answer(q, context)
        f1 = compute_f1(answers[i], pred)
        total_f1 += f1
        
        # 保存每个问题的详细结果
        results.append({
            "Question": q,
            "Predicted Answer": pred,
            "Gold Answer": answers[i],
            "F1": f1
        })
        
        print(f"\n[Q{i+1}]")
        print("Question:", q)
        print("Predicted Answer:", pred)

    # 计算平均值
    avg_f1 = total_f1 / len(questions)
    
    print(f"\n✅ Final Evaluation with {retriever_type.upper()} + {reader_type.upper()}:")
    print(f"Average F1: {avg_f1:.3f}")
    
    # 将结果保存到 DataFrame
    df = pd.DataFrame(results)
    
    # 保存到 Excel 文件（可选）
    df.to_excel("evaluation_results.xlsx", index=False)
    
    # 可视化 和 F1
    plot_metrics(df, avg_f1)

def plot_metrics(df, avg_f1):
    # 使用 Seaborn 进行可视化
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制 F1 的分布
    sns.histplot(df["F1"], kde=True, ax=ax[1], color='green')
    ax[1].set_title("F1 Score Distribution")
    ax[1].set_xlabel("F1 Score")
    ax[1].set_ylabel("Frequency")
    
    # 添加平均值线
    ax[1].axvline(avg_f1, color='red', linestyle='--', label=f"Avg F1: {avg_f1:.3f}")
    ax[1].legend()
    
     # 保存图像到文件
    plt.savefig(f"f1_distribution_{retriever_type}_{reader_type}.png")
    plt.close()

# ----- Run Example -----  
if __name__ == "__main__":
    for retriever_type in ["dense", "bm25", "hybrid"]: 
        for reader_type in ["hf"]:
            run_pipeline(retriever_type=retriever_type, reader_type=reader_type)