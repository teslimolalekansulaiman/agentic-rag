def build_models_response():
    """Build OpenAI-compatible models list response."""
    return {
        "object": "list",
        "data": [
            {
                "id": "crew-ai-rag",
                "object": "model",
                "created": 1677652288,
                "owned_by": "crew-ai-rag",
                "permission": [],
                "root": "crew-ai-rag",
                "parent": None,
                "max_tokens": 131072,
                "context_length": 131072
            }
        ]
    }


def build_chat_completion_response(
    session_id: str,
    model: str,
    answer: str,
    user_message: str
):
    """Build OpenAI-compatible chat completion response."""
    return {
        "id": f"chatcmpl-{session_id}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(user_message.split()),
            "completion_tokens": len(answer.split()),
            "total_tokens": len(user_message.split()) + len(answer.split())
        }
    }


def normalize_crew_output(result):
    """Normalize different crew output formats to string."""
    if hasattr(result, "raw"):
        return result.raw
    elif hasattr(result, "output"):
        return result.output
    else:
        return str(result)