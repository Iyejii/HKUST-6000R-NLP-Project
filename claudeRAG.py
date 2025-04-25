import json
import re
import time
import logging
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import Counter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness
)
from datasets import Dataset

# ------------------ Configuration ------------------
CLAUDE_API_BASE = "https://api.qingtian.shop/v1"
# CLAUDE_API_KEY = "sk-2E0zZGugGIITx5nYc4L2QBnpuewbwqlAfanpgJE8uQ1XkFdl"
CLAUDE_API_KEY = "sk-iQ5DLMedr9sH7EaE6iZ47Yq5EATymjpBKrfztPlrKwMVyt11"
SELECTED_MODEL = "claude-3-7-sonnet-20250219"
RETRIEVAL_TOP_K = 3
MAX_SAMPLES = 1000  # Reduced for testing, can increase to 1000

# ------------------ Logging Setup ------------------
logging.basicConfig(
    filename='rag_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ------------------ Data Loading ------------------
def load_pubmedqa_data(file_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Load the PubMedQA dataset from a JSON file."""
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


# ------------------ Retrieval Classes ------------------
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


class DenseRetriever:
    def __init__(self, corpus: list, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.corpus = corpus
        self.embeddings = self.model.encode(corpus, convert_to_tensor=True)

    def retrieve(self, query: str, top_k=3) -> list:
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=top_k)[0]
        return [self.corpus[hit['corpus_id']] for hit in hits]


class HybridRetriever:
    def __init__(self, corpus: list, bm25_weight=0.5, dense_weight=0.5,
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.bm25 = BM25Retriever(corpus)
        self.dense = DenseRetriever(corpus, model_name)
        self.corpus = corpus
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def retrieve(self, query: str, top_k=3) -> list:
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.bm25.get_scores(tokenized_query)

        query_embedding = self.dense.model.encode(query, convert_to_tensor=True)
        dense_scores = util.cos_sim(query_embedding, self.dense.embeddings)[0].cpu().tolist()

        hybrid_scores = [
            self.bm25_weight * bm25_scores[i] + self.dense_weight * dense_scores[i]
            for i in range(len(self.corpus))
        ]

        top_indices = sorted(range(len(hybrid_scores)), key=lambda i: hybrid_scores[i], reverse=True)[:top_k]
        return [self.corpus[i] for i in top_indices]


# ------------------ Claude API Client ------------------
class ClaudeAPIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def chat_completions_create(self, model, messages):
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.3
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            )
            if response.status_code == 429:
                backoff = 2 ** 0
                time.sleep(backoff)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"API call failed: {str(e)}")
            return None


# ------------------ Evaluation Framework ------------------
class RAGEvaluator:
    def __init__(self):
        self.metrics = {
            'answer_relevancy': answer_relevancy,
            'faithfulness': faithfulness,
            'context_precision': context_precision,
            'context_recall': context_recall,
            'answer_correctness': answer_correctness
        }

    def compute_f1(self, gold: str, pred: str) -> float:
        def normalize(text: str) -> List[str]:
            text = re.sub(r'[^\w\s]', '', text.lower()).split()
            return [word for word in text if word not in {'a', 'an', 'the'}]

        gold_tokens = normalize(gold)
        pred_tokens = normalize(pred)

        common = Counter(gold_tokens) & Counter(pred_tokens)
        num_same = sum(common.values())

        if not gold_tokens or not pred_tokens:
            return float(gold_tokens == pred_tokens)
        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * (precision * recall) / (precision + recall)

    def evaluate_ragas(self, questions: List[str], answers: List[str], contexts: List[List[str]],
                       ground_truths: List[str]) -> Dict:
        dataset = Dataset.from_dict({
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        })

        result = evaluate(dataset, metrics=list(self.metrics.values()))
        return result


# Function to generate an answer
def generate_answer(context, question, client, model_name):
    prompt = f"Answer the question based on the context below:\nContext: {context}\nQuestion: {question}\nAnswer:"
    try:
        completion = client.chat_completions_create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        if completion and completion.get('choices'):
            answer = completion['choices'][0]['message']['content'].strip()
            translation_prompt = f"Translate the following text to English:\n{answer}\nEnglish Translation:"
            translation_completion = client.chat_completions_create(
                model=model_name,
                messages=[{"role": "user", "content": translation_prompt}]
            )
            translated_answer = translation_completion['choices'][0]['message']['content'].strip()
            print(f"Question: {question}")
            print(f"Answer: {translated_answer}")
            return translated_answer
        else:
            print(f"Question: {question}")
            print("Answer: The model did not return a valid answer.")
            return "The model did not return a valid answer."
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        print(f"Question: {question}")
        print("Answer: Error generating answer.")
        return "Error generating answer."


# ------------------ Main Pipeline ------------------
def initialize_retriever(retriever_type: str, corpus: List[str]):
    """Initialize the specified retriever type."""
    if retriever_type == "bm25":
        return BM25Retriever(corpus)
    elif retriever_type == "dense":
        return DenseRetriever(corpus)
    elif retriever_type == "hybrid":
        return HybridRetriever(corpus)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def build_prompt(question: str, contexts: List[str]) -> str:
    context_str = "\n\n".join([f"[[Context {i + 1}]] {ctx}" for i, ctx in enumerate(contexts)])
    prompt = f"Answer the question based on the context below:\n\nContext: {context_str}\n\nQuestion: {question}\nAnswer:"
    return prompt


def run_evaluation_pipeline(retriever_type: str = "hybrid",
                            file_path: str = r"C:\Users\Laptop\Desktop\NLP\ori_pqau.json"):
    """Run the complete evaluation pipeline."""
    # Initialize components
    claude = ClaudeAPIClient(CLAUDE_API_BASE, CLAUDE_API_KEY)
    evaluator = RAGEvaluator()

    # Load data
    logging.info(f"Loading data from {file_path}")
    corpus, questions, gold_answers = load_pubmedqa_data(file_path)
    corpus = corpus[:MAX_SAMPLES]
    questions = questions[:MAX_SAMPLES]
    gold_answers = gold_answers[:MAX_SAMPLES]

    # Initialize retriever
    retriever = initialize_retriever(retriever_type, corpus)

    results = []
    rag_eval_data = {
        'questions': [],
        'answers': [],
        'contexts': [],
        'ground_truths': []
    }

    for idx in range(len(questions)):
        try:
            # Retrieval phase
            question = questions[idx]
            gold_answer = gold_answers[idx]
            start_time = time.time()
            retrieved_docs = retriever.retrieve(question)
            retrieval_time = time.time() - start_time

            # Merge the retrieved documents as the context
            context = "\n\n".join(retrieved_docs)

            # Call the new function to generate the answer
            answer = generate_answer(context, question, claude, SELECTED_MODEL)
            generation_time = time.time() - start_time - retrieval_time

            # Store for batch RAGAS evaluation
            rag_eval_data['questions'].append(question)
            rag_eval_data['answers'].append(answer)
            rag_eval_data['contexts'].append(retrieved_docs)
            rag_eval_data['ground_truths'].append(gold_answer)

            # Immediate F1 calculation
            metrics = {
                "f1": evaluator.compute_f1(gold_answer, answer),
                "retrieval_latency": retrieval_time,
                "generation_latency": generation_time
            }

            # Log results
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "question_id": idx,
                "question": question[:100] + "..." if len(question) > 100 else question,
                **metrics,
                "context_hashes": [hash(doc) for doc in retrieved_docs]
            }
            logging.info(json.dumps(log_entry))

            results.append(metrics)

        except Exception as e:
            logging.error(f"Error processing question {idx}: {str(e)}")
            continue

    # Batch RAGAS evaluation
    logging.info("Starting RAGAS evaluation")
    ragas_results = evaluator.evaluate_ragas(
        rag_eval_data['questions'],
        rag_eval_data['answers'],
        rag_eval_data['contexts'],
        rag_eval_data['ground_truths']
    )

    # Generate report
    generate_performance_report(results, ragas_results, retriever_type)


def generate_performance_report(results: List[Dict], ragas_results: Dict, retriever_type: str):
    """Generate a performance report and visualizations."""
    # Convert to pandas DataFrame
    df = pd.DataFrame(results)

    # Add RAGAS metrics
    for metric, scores in ragas_results.items():
        df[metric] = scores

    # Summary statistics
    report = {
        "retriever": retriever_type,
        "avg_f1": df["f1"].mean(),
        "avg_retrieval_latency": df["retrieval_latency"].mean(),
        "avg_generation_latency": df["generation_latency"].mean(),
        "avg_faithfulness": df["faithfulness"].mean(),
        "avg_answer_relevancy": df["answer_relevancy"].mean(),
        "avg_context_precision": df["context_precision"].mean(),
        "avg_answer_correctness": df["answer_correctness"].mean()
    }

    # print for debug
    for metric, value in report.items():
        print(f"Metric: {metric}, Value: {value}, Type: {type(value)}")

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[["f1", "faithfulness", "answer_relevancy", "answer_correctness"]])
    plt.title(f"Performance Metrics ({retriever_type.upper()} Retriever)")
    plt.savefig(f"metrics_{retriever_type}.png")
    plt.close()
    print("\n" + "=" * 40)
    print(f"EVALUATION REPORT - {retriever_type.upper()} RETRIEVER")
    print("=" * 40)
    for metric, value in report.items():
        if isinstance(value, (int, float)):
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value}")
    print("=" * 40)

#______________________for bm evalution______________________________
import json
import logging
import pandas as pd


def parse_logfile(log_path: str) -> pd.DataFrame:
    """从日志文件中提取已处理的问题结果"""
    results = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if " - INFO - " in line:
                # 提取JSON部分
                json_str = line.split(" - INFO - ")[1].strip()
                try:
                    entry = json.loads(json_str)
                    # 提取关键指标
                    results.append({
                        "question_id": entry["question_id"],
                        "f1": entry["f1"],
                        "retrieval_latency": entry["retrieval_latency"],
                        "generation_latency": entry["generation_latency"]
                    })
                except json.JSONDecodeError:
                    logging.warning(f"Invalid JSON in log line: {line}")

    return pd.DataFrame(results).sort_values("question_id")

def mock_ragas_results(num_samples: int) -> dict:
    """生成模拟的RAGAS评估结果（实际需替换为真实数据）"""
    return {
        "faithfulness": [0.85] * num_samples,
        "answer_relevancy": [0.82] * num_samples,
        "context_precision": [0.78] * num_samples,
        "answer_correctness": [0.80] * num_samples
    }
def run_log_based_evaluation(log_path: str, retriever_type: str):
    """基于日志文件的评估报告生成"""
    # 1. 解析日志获取结果
    df_log = parse_logfile(log_path)
    results = df_log.to_dict("records")

    # 2. 模拟/加载RAGAS结果（需替换为真实数据）
    num_samples = len(results)
    ragas_results = mock_ragas_results(num_samples)  # 实际需替换为真实评估数据

    # 3. 转换为DataFrame并补充RAGAS指标
    df = pd.DataFrame(results)
    for metric, scores in ragas_results.items():
        df[metric] = scores  # 先添加列
        df[metric] = pd.to_numeric(df[metric], errors='coerce')  # 确保数据为数值类型

    print("DataFrame columns:", df.columns)  # 打印列名进行调试

    # 4. 生成报告
    generate_performance_report(df, ragas_results, retriever_type)

if __name__ == "__main__":
    LOG_FILE = "rag_evaluation_bak.log"  # 你的日志文件路径
    RETRIEVER_TYPE = "bm25"  # 日志对应的检索器类型

    # 直接基于日志生成报告
    run_log_based_evaluation(LOG_FILE, RETRIEVER_TYPE)

# ------------------ Execution ------------------
if __name__ == "__main__":
    TEST_FILE = r"C:\Users\Laptop\Desktop\NLP\ori_pqau.json"  # Replace with your test file
    for retriever in ["bm25", "dense", "hybrid"]:
        logging.info(f"Starting evaluation with {retriever} retriever")
        try:
            run_evaluation_pipeline(retriever, TEST_FILE)
        except Exception as e:
            logging.error(f"Pipeline failed for {retriever}: {str(e)}")
            continue