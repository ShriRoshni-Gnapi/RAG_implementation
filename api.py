"""
FastAPI Server Gateway for PT-RAG (Production Deployment).

Provides:
  - OpenAI-compatible endpoint: POST /v1/chat/completions (port 8080)
  - REST endpoint: POST /ask
  - Textbook references for a SOAP note: POST /references
  - Health check: GET /health

Usage:
  uvicorn api:app --host 0.0.0.0 --port 8080
"""

import time
import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from query_engine import PTQueryEngine
from llm_client import LLMClient, LLMError
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


# --------------------------------------------------------------------------
# POST /references: textbook references for the S, O and A of a saved SOAP note
# --------------------------------------------------------------------------

MAX_SECTION_CHARS = 8000          # per section; a longer note is refused, not cut


class ReferencesRequest(BaseModel):
    subjective: str = Field("", max_length=MAX_SECTION_CHARS)
    objective: str = Field("", max_length=MAX_SECTION_CHARS)
    assessment: str = Field("", max_length=MAX_SECTION_CHARS)
    top_k: int = Field(5, ge=1, le=20)


class ReferenceItem(BaseModel):
    book: str
    author: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    pages: Optional[str] = None
    summary: List[str]            # checked, model-written paragraphs (the only model text)
    sentences: List[str]          # the book's own sentences the summary is about


class ReferencesResponse(BaseModel):
    status: str                   # "ready", or "empty" when nothing relevant had a supported summary
    generator: str                # which prompts and filters produced this set
    references: List[ReferenceItem]


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError):
    """
    The default 422 body echoes the submitted values back. On /references those
    are a patient's note, so that route answers a generic message; every other
    route keeps FastAPI's default answer.
    """
    if request.url.path == "/references":
        return JSONResponse(status_code=422, content={"detail": "The request is invalid."})
    return await request_validation_exception_handler(request, exc)


def references_response(result: Dict[str, Any]) -> ReferencesResponse:
    """
    The caller's view of `find_references`: only citations that have a
    supported summary, with the fields below. Never the model's raw reply, the
    candidate list, or the notes and passage used to check the summaries.
    """
    if any(n["reason"].startswith("summary call failed") for n in result["summary_notes"]):
        raise HTTPException(status_code=503, detail="The reference service is unavailable (summary).")
    items = [
        ReferenceItem(
            book=e["book"], author=e["author"], chapter=e["chapter"], section=e["section"],
            pages=e["pages"], summary=e["summary"], sentences=e["sentences"],
        )
        for e in result["reference_entries"] if e.get("summary")
    ]
    return ReferencesResponse(
        status="ready" if items else "empty",
        generator=result["generator"],
        references=items,
    )


@app.post("/references", response_model=ReferencesResponse)
def references(req: ReferencesRequest):
    """
    Textbook references, with a checked summary of each cited passage, for the
    Subjective, Objective and Assessment text of a note. Nothing else about the
    patient is accepted. The note is never logged or echoed in an error.
    """
    sections = [
        (label, text) for label, text in
        (("S", req.subjective), ("O", req.objective), ("A", req.assessment)) if text.strip()
    ]
    if not sections:
        raise HTTPException(status_code=400, detail="No Subjective, Objective or Assessment text was given.")
    try:
        result = engine.find_references(
            "", top_k=req.top_k, summaries=True, summary_only=True, sections=sections)
    except (LLMError, qdrant_store.VectorStoreError) as e:
        raise HTTPException(status_code=503, detail=f"The reference service is unavailable ({type(e).__name__}).")
    except Exception as e:
        print(f"/references failed: {type(e).__name__}", flush=True)      # the type only, never the message
        raise HTTPException(status_code=500, detail="The reference lookup failed.")
    return references_response(result)


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

