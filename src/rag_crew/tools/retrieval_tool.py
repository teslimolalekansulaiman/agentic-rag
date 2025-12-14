from crewai.tools import tool
from src.rag.query_engine import RAGQueryEngine
from typing import Dict, Union, Any

_query_engine_instance = None

def get_query_engine():
    """Returns a single persistent RAGQueryEngine instance"""
    global _query_engine_instance
    if _query_engine_instance is None:
        _query_engine_instance = RAGQueryEngine()
    return _query_engine_instance

@tool("Document Retrieval Tool")
def document_retrieval_tool(question: Any) -> str:
    """Retrieves relevant context from a collection of policy and standards documents."""

    if isinstance(question, dict):
        question = question.get("description") or str(question)
    elif not isinstance(question, str):
        question = str(question)

    # ✅ Use persistent engine instead of creating new one
    qe = get_query_engine()
    res = qe.chat(question)

    context = "## Retrieved Context\n\n"
    context += str(res)

    # Safely access chat history
    if hasattr(qe, 'chat_engine') and hasattr(qe.chat_engine, 'chat_history'):
        chat_history = qe.chat_engine.chat_history

        if chat_history and len(chat_history) > 0:
            context += "\n\n## Last Conversation\n\n"
            last_msg = chat_history[-1]

            # Access ChatMessage properties
            role = getattr(last_msg, 'role', 'unknown')
            content = getattr(last_msg, 'content', str(last_msg))

            context += f"**{role.title()}**: {content}\n"

    return context



