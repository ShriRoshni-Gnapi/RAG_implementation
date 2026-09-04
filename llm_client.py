"""
LLM Inference Client (OpenAI-compatible HTTP API for NVIDIA Nemotron / vLLM / MedGemma).

Provides robust chat completion generation for LLM models served via vLLM,
Ollama, llama.cpp, TGI, or OpenAI-compatible proxies.

Features:
  * Zero extra dependencies (uses standard library urllib)
  * Supports NVIDIA Nemotron, MedGemma, and other OpenAI-compatible chat endpoints
  * Intelligent endpoint resolution (handles base URL, /v1, and /chat/completions)
  * Cleans reasoning tokens (<think>...</think>) if emitted by reasoning models
  * Diagnostic connection test with latency measurement
  * Actionable exception reporting
"""

import json
import re
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Generator, Any

from config import (
    LLM_API_URL,
    LLM_API_TOKEN,
    LLM_MODEL,
    LLM_TIMEOUT,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)


class LLMError(RuntimeError):
    """Base exception for LLM inference failures."""


class LLMConnectionError(LLMError):
    """Failed to reach the inference endpoint."""


class LLMAuthError(LLMError):
    """Authentication failed (HTTP 401 / 403)."""


class LLMTimeoutError(LLMError):
    """Inference request timed out."""


def _normalize_endpoint(url: str) -> str:
    """Ensure the URL points directly to the /chat/completions endpoint."""
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def _clean_reasoning_tags(text: str) -> str:
    """
    Remove <think>...</think> reasoning traces if present in reasoning models
    like Nemotron-Reasoning / DeepSeek-R1 so the final clinical answer is clean.
    """
    if not text:
        return ""
    # Strip paired <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # If an opening tag was omitted by the tokenizer/vLLM but closing </think> is present,
    # the text before </think> is internal reasoning
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    # Strip any lingering leading/trailing whitespace
    return cleaned.strip()



class LLMClient:
    """
    OpenAI-compatible HTTP client for LLM chat completions (NVIDIA Nemotron, MedGemma, etc.).
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        raw_url = api_url or LLM_API_URL
        self.endpoint_url = _normalize_endpoint(raw_url)
        self.base_url = raw_url.rstrip("/")
        self.api_token = api_token if api_token is not None else LLM_API_TOKEN
        self.model = model or LLM_MODEL
        self.timeout = timeout if timeout is not None else LLM_TIMEOUT
        self.temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "PT-RAG-LLMClient/1.0",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a synchronous non-streaming chat completion request.
        """
        if messages is None:
            msg_list = []
            if system_prompt:
                msg_list.append({"role": "system", "content": system_prompt})
            msg_list.append({"role": "user", "content": user_prompt})
        else:
            msg_list = messages

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": msg_list,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop

        body_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=body_bytes,
            headers=self._get_headers(),
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status_code = resp.getcode()
                raw_body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            if e.code in (401, 403):
                raise LLMAuthError(
                    f"Authentication failed (HTTP {e.code}) for {self.endpoint_url}.\n"
                    f"Details: {err_body}\n"
                    f"Check NVIDIA_API_TOKEN / LLM_API_KEY in .env."
                ) from e
            raise LLMError(
                f"LLM API request failed with HTTP {e.code}: {e.reason}\n"
                f"Endpoint: {self.endpoint_url}\n"
                f"Response: {err_body}"
            ) from e
        except urllib.error.URLError as e:
            if "timed out" in str(e.reason).lower():
                raise LLMTimeoutError(
                    f"Inference request timed out after {self.timeout}s.\n"
                    f"Endpoint: {self.endpoint_url}"
                ) from e
            raise LLMConnectionError(
                f"Failed to connect to LLM server at {self.endpoint_url}.\n"
                f"Reason: {e.reason}\n"
                f"Ensure the server is running and accessible."
            ) from e
        except Exception as e:
            if "timed out" in str(e).lower():
                raise LLMTimeoutError(
                    f"Inference request timed out after {self.timeout}s."
                ) from e
            raise LLMError(f"Unexpected error calling LLM API: {e}") from e

        latency_ms = (time.time() - t0) * 1000

        try:
            data = json.loads(raw_body)
        except json.JSONDecodeError as e:
            raise LLMError(
                f"Invalid JSON returned by LLM endpoint ({self.endpoint_url}):\n{raw_body[:300]}"
            ) from e

        choices = data.get("choices", [])
        if not choices:
            raise LLMError(f"No choices returned in API response: {data}")

        choice = choices[0]
        message = choice.get("message", {})
        raw_text = message.get("content", "") or ""
        reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""

        clean_text = _clean_reasoning_tags(raw_text)

        return {
            "text": clean_text,
            "raw_text": raw_text,
            "reasoning": reasoning_content,
            "model": data.get("model", self.model),
            "usage": data.get("usage", {}),
            "latency_ms": latency_ms,
            "finish_reason": choice.get("finish_reason"),
        }

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """
        Stream generation tokens.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }

        body_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=body_bytes,
            headers=self._get_headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or line_str.startswith(":"):
                        continue
                    if line_str.startswith("data: "):
                        data_part = line_str[6:]
                        if data_part == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_part)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except Exception:
            # Fallback to non-streaming if stream mode encountered connection error
            res = self.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            yield res["text"]

    def check_connection(self) -> Dict[str, Any]:
        """
        Send a lightweight ping to verify that the server is reachable and responsive.
        """
        t0 = time.time()
        res = self.generate(
            system_prompt="You are a medical reference assistant.",
            user_prompt="Reply with the exact word 'READY' and nothing else.",
            max_tokens=20,
            temperature=0.0,
        )
        latency = (time.time() - t0) * 1000
        return {
            "status": "ok",
            "latency_ms": latency,
            "model": res.get("model", self.model),
            "response": res.get("text", "").strip(),
        }


def get_llm_client(**kwargs) -> LLMClient:
    """Convenience factory function."""
    return LLMClient(**kwargs)


# Backwards compatibility alias
MedGemmaClient = LLMClient
get_medgemma_client = get_llm_client
MedGemmaError = LLMError
MedGemmaConnectionError = LLMConnectionError
MedGemmaAuthError = LLMAuthError
MedGemmaTimeoutError = LLMTimeoutError


if __name__ == "__main__":
    import sys
    print("Testing LLM Client connection...")
    print(f"URL:   {LLM_API_URL}")
    print(f"Model: {LLM_MODEL}")
    print(f"Token: {'[CONFIGURED]' if LLM_API_TOKEN else '[NOT SET]'}")

    client = LLMClient()
    try:
        chk = client.check_connection()
        print(f"\n[OK] LLM server responsive ({chk['latency_ms']:.0f} ms)")
        print(f"Model reported: {chk['model']}")
        print(f"Response: {chk['response']}")
    except Exception as e:
        print(f"\n[FAIL] LLM check failed: {type(e).__name__}: {e}")
        sys.exit(1)
