"""
FastAPI Server Gateway for PT-RAG (Production Deployment).

Provides:
  - OpenAI-compatible endpoint: POST /v1/chat/completions (port 8080)
  - REST endpoint: POST /ask
  - Health check: GET /health

Usage:
  uvicorn api:app --host 0.0.0.0 --port 8080
"""

import time
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from query_engine import PTQueryEngine
from llm_client import LLMClient
from config import LLM_MODEL, QDRANT_URL, LLM_API_URL
import qdrant_store

app = FastAPI(
    title="PT-RAG Clinical Assistant Gateway",
    description="Physiotherapy RAG + NVIDIA Nemotron / MedGemma Inference Gateway",
    version="1.0.0",
)

# Enable CORS for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global query engine instance (lazy loads embedding model & connects to Qdrant)
engine = PTQueryEngine()


class AskRequest(BaseModel):
    question: str = Field(..., description="The clinical physiotherapy question to answer")
    book_key: Optional[str] = Field(None, description="Optional specific book_id to restrict search")
    top_k: int = Field(5, description="Number of textbook chunks to retrieve", ge=1, le=20)


class AskResponse(BaseModel):
    question: str
    answer: str
    formatted_output: str
    citations: List[Dict[str, Any]]
    model: str
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = 5


@app.on_event("startup")
def startup_event():
    """Pre-load embedding model on startup so first request has zero cold-start delay."""
    try:
        from query_engine import get_embedding_model
        get_embedding_model()
    except Exception as e:
        print(f"Warning: Failed to pre-load embedding model on startup: {e}")


@app.get("/health")
def health_check():
    """Check connectivity to Qdrant vector DB and NVIDIA LLM server."""
    qdrant_status = "error"
    qdrant_points = 0
    try:
        client = qdrant_store.get_client()
        qdrant_points = qdrant_store.count_points(client, qdrant_store.QDRANT_COLLECTION)
        qdrant_status = f"connected ({qdrant_points:,} points in '{qdrant_store.QDRANT_COLLECTION}')"
        client.close()
    except Exception as e:
        qdrant_status = f"unreachable: {e}"

    llm_status = "error"
    try:
        llm = LLMClient()
        conn = llm.check_connection()
        llm_status = f"connected ({conn['latency_ms']:.0f}ms latency)"
    except Exception as e:
        llm_status = f"unreachable: {e}"

    is_healthy = "connected" in qdrant_status and "connected" in llm_status
    return {
        "status": "healthy" if is_healthy else "degraded",
        "qdrant": {
            "url": QDRANT_URL,
            "collection": qdrant_store.QDRANT_COLLECTION,
            "total_points": qdrant_points,
            "status": qdrant_status,
        },
        "llm_server": {
            "endpoint": LLM_API_URL,
            "model": LLM_MODEL,
            "status": llm_status,
        },
    }


@app.post("/ask", response_model=AskResponse)
def ask_pt(req: AskRequest):
    """
    Main PT-RAG endpoint. Retrieves textbook excerpts, synthesizes with LLM, and returns clinical answer.
    """
    try:
        res = engine.query(
            question=req.question,
            book_key=req.book_key,
            all_books=(req.book_key is None),
            top_k=req.top_k,
        )
        return AskResponse(
            question=res["question"],
            answer=res["answer"],
            formatted_output=res["formatted_output"],
            citations=res["all_retrieved"],
            model=res["model"],
            retrieval_ms=res["retrieval_ms"],
            generation_ms=res["generation_ms"],
            total_ms=res["total_ms"],
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")


@app.post("/v1/chat/completions")
def openai_chat_completions(req: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint for drop-in use with OpenAI SDK, LangChain, and chat UIs.
    """
    if not req.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty.")

    # Get the latest user message
    user_query = ""
    for msg in reversed(req.messages):
        if msg.role == "user":
            user_query = msg.content
            break

    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in request.")

    try:
        res = engine.query(
            question=user_query,
            top_k=req.top_k or 5,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        full_content = res["formatted_output"]
        created_ts = int(time.time())
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": res["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)

