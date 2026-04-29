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


async def vllm_chat(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    response = await client.chat.completions.create(
        model=model or VLLM_MODEL,
        messages=messages,
        max_tokens=max_tokens or MAX_COMPLETION_TOKENS,
        temperature=VLLM_TEMPERATURE if temperature is None else temperature,
    )

    return str(response.choices[0].message.content or "").strip()
