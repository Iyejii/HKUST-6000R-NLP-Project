# HKUST-6000R-NLP-Project
Repo for HKUST-6000R-NLP group project

# Comparative Analysis of Retrieval Algorithms and LLM Combinations in Retrieval-Augmented Generation (RAG) Systems

## 1. Task Introduction
  Retrieval-Augmented Generation (RAG) enhances answer accuracy by integrating retrieval systems and generative LLMs to ground responses in external knowledge. This project evaluates: (1) retrieval methods—BM25 (keyword-based), BGE-M3 (semantic embeddings), and hybrid Reciprocal Rank Fusion (RRF); (2) generation with LLMs for synthesizing answers from retrieved context.
## 2. Importance of the Task
  This work addresses critical RAG challenges: (1) improving factual accuracy by grounding LLMs in evidence to reduce hallucinations; (2) balancing speed (sparse retrieval) and precision (dense/hybrid methods); (3) enabling reliable applications in high-stakes domains like medical, legal, and customer support QA.
## 3. Proposed Methodologies
### 3.1 Data Preparation: 
  Preprocess documents into 200–512 token chunks with overlap; standardize normalization/augmentation
### 3.2 Retrieval Algorithm Comparison:
  Test three retrieval strategies with pros/cons: 
  1.	Sparse Retrieval: BM25 (keyword matching, high speed, low semantic understanding).
  2.	Dense Retrieval: BGE-M3 (semantic embeddings, supports multi-lingual and hybrid search).
  3.	Hybrid Retrieval: Combine BM25 and BGE-M3 scores via Reciprocal Rank Fusion (RRF) for balanced precision.
### 3.3 LLM Generation
  1.	Baseline Models:To be determined based on task requirements and available resources.
  2.	Prompt Engineering: Use chain-of-thought prompts (e.g., "Let’s think step by step") to enhance reasoning, Inject retrieved contexts into system prompts (e.g., "Answer using ONLY the following information: {context}").
### 3.4 Evaluation Framework
    Evaluation combines automated metrics—RAGAS (assessing faithfulness, context precision, and relevance via GPT-4-as-judge) and FactScore (quantifying factual consistency against references)—with human-annotated scores from 100 samples for accuracy, fluency, and completeness to validate system reliability

