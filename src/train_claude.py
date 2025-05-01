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
from config import OPENAI_API_KEY, OPENAI_API_BASE_URL
from openai import OpenAI
import os

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
    def __init__(self, corpus: list, bm25_weight=0.5, dense_weight=0.5,
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
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

# ----- OpenAIReader -----
class OpenAIReader:
    def __init__(self, model_name="claude-3-7-sonnet-20250219", base_url=None):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_API_BASE_URL
        )

    def answer(self, question: str, context: str) -> str:
        prompt = f"Answer the question based on the context below:\n\nContext: {context}\n\nQuestion: {question}\nAnswer in english:"
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answer = completion.choices[0].message.content.strip()
        return answer

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
def run_pipeline(retriever_type="dense", reader_type="openai"):
    # 加载 PubMedQA 数据集
    pubmedqa_file_path = r'C:\Users\Laptop\Desktop\NLP\ori_pqau.json'  # 替换PubMedQA 数据文件路径
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
    if reader_type == "hf":
        reader = HFReader()
    elif reader_type == "openai":
        reader = OpenAIReader(model_name="claude-3-7-sonnet-20250219")
    else:
        raise ValueError(f"Unknown reader type: {reader_type}")

    # 初始化一个列表来保存结果
    results = []

    # 创建临时文件以不断保存进度
    temp_file = f"temp_results_{retriever_type}_{reader_type}.csv"

    # 如果已存在临时文件，则加载已有结果
    if os.path.exists(temp_file):
        try:
            temp_df = pd.read_csv(temp_file)
            start_idx = len(temp_df)
            results = temp_df.to_dict('records')
            total_f1 = temp_df['F1'].sum()
            print(f"已加载 {start_idx} 条已完成的结果")
        except Exception as e:
            print(f"加载临时文件失败: {e}")
            start_idx = 0
            total_f1 = 0
    else:
        start_idx = 0
        total_f1 = 0

    for i, q in enumerate(questions[start_idx:], start=start_idx):
        try:
            context = " ".join(retriever.retrieve(q, top_k=3))
            pred = reader.answer(q, context)
            f1 = compute_f1(answers[i], pred)
            total_f1 += f1

            # 保存每个问题的详细结果
            result = {
                "Question": q,
                "Predicted Answer": pred,
                "Gold Answer": answers[i],
                "F1": f1
            }
            results.append(result)

            print(f"\n[Q{i + 1}]")
            print("Question:", q)
            print("Predicted Answer:", pred)

            # 每处理 5 个问题就保存一次临时文件
            if (i + 1) % 5 == 0 or i == len(questions) - 1:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(temp_file, index=False)
                print(f"进度已保存至 {temp_file}, 当前处理了 {i + 1}/{len(questions)} 个问题")

        except Exception as e:
            print(f"处理问题 {i + 1} 时出错: {e}")
            # 保存当前进度
            temp_df = pd.DataFrame(results)
            temp_df.to_csv(temp_file, index=False)
            print(f"错误已记录，进度已保存至 {temp_file}")

    # 计算平均值
    avg_f1 = total_f1 / len(questions)

    print(f"\n✅ Final Evaluation with {retriever_type.upper()} + {reader_type.upper()}:")
    print(f"Average F1: {avg_f1:.3f}")

    # 将结果保存到 DataFrame
    df = pd.DataFrame(results)

    # 保存到 Excel 文件
    final_file = f"evaluation_results_{retriever_type}_{reader_type}.xlsx"
    df.to_excel(final_file, index=False)
    print(f"最终结果已保存至 {final_file}")

    # 可视化和 F1
    plot_metrics(df, avg_f1, retriever_type, reader_type)


def plot_metrics(df, avg_f1, retriever_type, reader_type):
    # 使用 Seaborn 进行可视化
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制 F1 的分布
    sns.histplot(df["F1"], kde=True, ax=ax[1], color='green')
    ax[1].set_title(f"F1 Score Distribution ({retriever_type}_{reader_type})")
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
        for reader_type in ["openai"]:
            run_pipeline(retriever_type=retriever_type, reader_type="openai")