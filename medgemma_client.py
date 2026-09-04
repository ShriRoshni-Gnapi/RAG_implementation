"""
Backwards-compatible wrapper for medgemma_client.py.
Redirects to llm_client.py.
"""

from llm_client import (
    LLMError as MedGemmaError,
    LLMConnectionError as MedGemmaConnectionError,
    LLMAuthError as MedGemmaAuthError,
    LLMTimeoutError as MedGemmaTimeoutError,
    LLMClient as MedGemmaClient,
    get_llm_client as get_medgemma_client,
    LLMError,
    LLMConnectionError,
    LLMAuthError,
    LLMTimeoutError,
    LLMClient,
    get_llm_client,
)

__all__ = [
    "MedGemmaError",
    "MedGemmaConnectionError",
    "MedGemmaAuthError",
    "MedGemmaTimeoutError",
    "MedGemmaClient",
    "get_medgemma_client",
    "LLMError",
    "LLMConnectionError",
    "LLMAuthError",
    "LLMTimeoutError",
    "LLMClient",
    "get_llm_client",
]

if __name__ == "__main__":
    from llm_client import main
    import sys
    # Run test
    client = MedGemmaClient()
    chk = client.check_connection()
    print(f"[OK] Inference Server responsive ({chk['latency_ms']:.0f} ms): {chk['model']}")
