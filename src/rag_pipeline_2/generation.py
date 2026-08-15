import json
from collections.abc import Iterator

import httpx

from rag_pipeline_2.schemas import ChatMessage

VLLM_BASE_URL = "http://127.0.0.1:8001"
VLLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 512


def _build_request_payload(messages: list[ChatMessage], max_new_tokens: int, stream: bool) -> dict:
    return {
        "model": VLLM_MODEL_NAME,
        "messages": [message.model_dump() for message in messages],
        "max_tokens": max_new_tokens,
        "stream": stream,
    }


def generate_response(
    messages: list[ChatMessage],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, int, int]:
    payload = _build_request_payload(messages, max_new_tokens, stream=False)
    response = httpx.post(f"{VLLM_BASE_URL}/v1/chat/completions", json=payload, timeout=120.0)
    response.raise_for_status()
    response_body = response.json()

    response_text = response_body["choices"][0]["message"]["content"]
    prompt_tokens = response_body["usage"]["prompt_tokens"]
    completion_tokens = response_body["usage"]["completion_tokens"]

    return response_text, prompt_tokens, completion_tokens


def stream_response(
    messages: list[ChatMessage],
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Iterator[str]:
    payload = _build_request_payload(messages, max_new_tokens, stream=True)

    with httpx.stream("POST", f"{VLLM_BASE_URL}/v1/chat/completions", json=payload, timeout=120.0) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue

            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break

            chunk = json.loads(data)
            delta_content = chunk["choices"][0]["delta"].get("content")
            if delta_content:
                yield delta_content