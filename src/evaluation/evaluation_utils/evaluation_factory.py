import os
import pandas as pd
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

def evaluate_single_question(question, answer, ground_truth, llm, embeddings):
    try:
        record = {
            "question": question,
            "answer": answer,
            "contexts": [answer],
            "ground_truth": ground_truth,
        }
        ds = Dataset.from_pandas(pd.DataFrame([record]))
        report = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
            llm=llm,
            embeddings=embeddings,
        )
        row = report.to_pandas().iloc[0]
        return {
            "faithfulness": row["faithfulness"],
            "answer_relevancy": row["answer_relevancy"],
            "context_recall": row["context_recall"],
            "context_precision": row["context_precision"],
        }
    except Exception as e:
        print(f"⚠️ Eval error: {e}")
        return {k: None for k in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]}

def get_llm():
    model = os.getenv("LLM_MODEL", "gemma3:4b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"🦙 LLM: {model} @ {base_url}")
    return ChatOllama(model=model, temperature=0, base_url=base_url, num_ctx=4096)


def get_embeddings():
    model = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text:v1.5")
    print(f"🔎 Embeddings: {model}")
    return OllamaEmbeddings(model=model)

