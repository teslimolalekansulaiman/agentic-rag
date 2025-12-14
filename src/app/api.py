import uvicorn
from fastapi import FastAPI
import asyncio
from pydantic import BaseModel
from typing import List, Optional

from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from src.app.models import ChatRequest
from src.app.responses import normalize_crew_output, build_chat_completion_response, build_models_response
from src.observability.phoenix_instrumentation import setup_phoenix_tracing
from src.rag_crew.orchestrator import run
from src.rag.query_engine import RAGQueryEngine

setup_phoenix_tracing()

app = FastAPI(title="Document RAG API")
# Add CORS middleware to allow requests from OpenWebUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/v1/models")
def list_models():
    return build_models_response()


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    try:
        # Validation already done by Pydantic!
        user_message = request.user_message

        # Run crew pipeline
        result = await asyncio.to_thread(run, user_message)
        answer = normalize_crew_output(result)

        # Build response
        return build_chat_completion_response(
            session_id=request.session_id,
            model=request.model,
            answer=answer,
            user_message=user_message
        )

    except Exception as e:
        return build_chat_completion_response(
            session_id=request.session_id,
            model=request.model,
            answer=f"⚠️ Sorry, something went wrong: {str(e)}",
            user_message=request.user_message
        )


if __name__ == "__main__":
    # This allows you to run the API server directly for testing
    uvicorn.run(app, host="0.0.0.0", port=8000)
