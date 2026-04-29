from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from .config import (
    MAX_COMPLETION_TOKENS,
    REQUEST_TIMEOUT_SECONDS,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
    VLLM_TEMPERATURE,
)


client = AsyncOpenAI(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
    timeout=REQUEST_TIMEOUT_SECONDS,
)


async def vllm_chat_completion(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Any:
    request_kwargs: dict[str, Any] = {
        "model": model or VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens or MAX_COMPLETION_TOKENS,
        "temperature": VLLM_TEMPERATURE if temperature is None else temperature,
    }

    if tools:
        request_kwargs["tools"] = tools
    if tool_choice is not None:
        request_kwargs["tool_choice"] = tool_choice

    return await client.chat.completions.create(**request_kwargs)
