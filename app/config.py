from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BACKEND_DIR / "config.json"
TEMPLATES_DIR = BACKEND_DIR / "templates"
SUBMISSIONS_DIR = BACKEND_DIR / "submissions"

DEFAULT_CONFIG = {
    "server": {
        "host": "0.0.0.0",
        "port": 8001,
    },
    "chat": {
        "systemPrompt": (
            "You are IDWay Assist, a document-aware service application agent. "
            "Your job is to help fill one service form at a time. "
            "Always identify the target service first, then extract only grounded facts from the user message and uploaded documents. "
            "Never invent field values. If a field is not supported by user-provided text or documents, leave it empty."
        ),
        "assistantStylePrompt": (
            "Ask one concise follow-up question at a time. "
            "When documents reveal form fields, silently use them instead of asking again. "
            "When every required field is complete, return a brief completion message."
        ),
    },
    "vllm": {
        "baseUrl": "http://localhost:8000/v1",
        "apiKey": "none",
        "model": "Qwen/Qwen3.5-9B",
        "requestTimeoutSeconds": 120,
        "maxCompletionTokens": 2048,
        "temperature": 0.2,
        "pdfVisionMaxPages": 3,
        "pdfTextMinChars": 80,
    },
}


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG

    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        data = json.load(config_file)

    if not isinstance(data, dict):
        return DEFAULT_CONFIG

    return data


CONFIG = load_config()
CHAT_CONFIG = CONFIG.get("chat", {})
VLLM_CONFIG = CONFIG.get("vllm", {})

SYSTEM_PROMPT = str(
    CHAT_CONFIG.get(
        "systemPrompt",
        DEFAULT_CONFIG["chat"]["systemPrompt"],
    )
)
ASSISTANT_STYLE_PROMPT = str(
    CHAT_CONFIG.get(
        "assistantStylePrompt",
        DEFAULT_CONFIG["chat"]["assistantStylePrompt"],
    )
)

VLLM_BASE_URL = os.getenv(
    "VLLM_BASE_URL",
    str(VLLM_CONFIG.get("baseUrl", DEFAULT_CONFIG["vllm"]["baseUrl"])),
)
VLLM_API_KEY = os.getenv(
    "VLLM_API_KEY",
    str(VLLM_CONFIG.get("apiKey", DEFAULT_CONFIG["vllm"]["apiKey"])),
)
VLLM_MODEL = os.getenv(
    "VLLM_MODEL",
    str(VLLM_CONFIG.get("model", DEFAULT_CONFIG["vllm"]["model"])),
)
REQUEST_TIMEOUT_SECONDS = float(
    os.getenv(
        "VLLM_TIMEOUT_SECONDS",
        str(
            VLLM_CONFIG.get(
                "requestTimeoutSeconds",
                DEFAULT_CONFIG["vllm"]["requestTimeoutSeconds"],
            )
        ),
    )
)
MAX_COMPLETION_TOKENS = int(
    os.getenv(
        "VLLM_MAX_COMPLETION_TOKENS",
        str(
            VLLM_CONFIG.get(
                "maxCompletionTokens",
                DEFAULT_CONFIG["vllm"]["maxCompletionTokens"],
            )
        ),
    )
)
VLLM_TEMPERATURE = float(
    os.getenv(
        "VLLM_TEMPERATURE",
        str(
            VLLM_CONFIG.get(
                "temperature",
                DEFAULT_CONFIG["vllm"]["temperature"],
            )
        ),
    )
)
PDF_VISION_MAX_PAGES = int(
    os.getenv(
        "PDF_VISION_MAX_PAGES",
        str(
            VLLM_CONFIG.get(
                "pdfVisionMaxPages",
                DEFAULT_CONFIG["vllm"]["pdfVisionMaxPages"],
            )
        ),
    )
)
PDF_TEXT_MIN_CHARS = int(
    os.getenv(
        "PDF_TEXT_MIN_CHARS",
        str(
            VLLM_CONFIG.get(
                "pdfTextMinChars",
                DEFAULT_CONFIG["vllm"]["pdfTextMinChars"],
            )
        ),
    )
)
