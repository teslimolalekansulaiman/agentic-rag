from llama_index.llms.ollama import Ollama
import os


def get_llm():
    """Return an LLM instance based on settings."""
    return Ollama("gemma3:4b", base_url= os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), temperature=0.2)
