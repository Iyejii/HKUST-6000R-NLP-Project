import json
import logging
import pandas as pd

from RAGClaude import generate_performance_report


def parse_logfile(log_path: str) -> pd.DataFrame:
    results = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if " - INFO - " in line:
                json_str = line.split(" - INFO - ")[1].strip()
                try:
                    entry = json.loads(json_str)
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
    return {
        "faithfulness": [0.85] * num_samples,
        "answer_relevancy": [0.82] * num_samples,
        "context_precision": [0.78] * num_samples,
        "answer_correctness": [0.80] * num_samples
    }
def run_log_based_evaluation(log_path: str, retriever_type: str = "bm25"):
    df_log = parse_logfile(log_path)
    results = df_log.to_dict("records")

    num_samples = len(results)
    # it's the mock value, to be completed
    ragas_results = mock_ragas_results(num_samples)

    df = pd.DataFrame(results)
    for metric, scores in ragas_results.items():
        df[metric] = scores
        df[metric] = pd.to_numeric(df[metric], errors='coerce')

    print("DataFrame columns:", df.columns)

    generate_performance_report(df, ragas_results, retriever_type)

if __name__ == "__main__":
    LOG_FILE = "rag_evaluation_bm_bak.log"
    RETRIEVER_TYPE = "bm25"
    run_log_based_evaluation(LOG_FILE, RETRIEVER_TYPE)