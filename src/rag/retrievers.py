from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank


def build_retriever(index, top_k=10):
    return VectorIndexRetriever(index=index, similarity_top_k=top_k)


def build_reranker(top_n= 5):
    return SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_n)


