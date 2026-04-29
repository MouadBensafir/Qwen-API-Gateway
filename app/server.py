from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile

from .config import (
    ASSISTANT_STYLE_PROMPT,
    MAX_TOOL_ROUNDS,
    OLLAMA_MODEL,
    RECENT_MESSAGE_COUNT,
    SUBMISSIONS_DIR,
    SYSTEM_PROMPT,
    TEMPLATES_DIR,
)
from .document_utils import build_document_payload
from .models import DeleteSessionResponse, PromptRequest, PromptResponse
from .ollama_client import ollama_chat_completion
from .session_store import (
    SessionState,
    delete_session,
    get_or_create_session,
    get_session_count,
    update_session_state,
)


app = FastAPI(title="IDWay Assist Agent", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SERVICE_CATALOG = {
    "ID Renewal": {
        "template": "id_renewal.json",
        "description": "Renew an existing ID card by confirming personal details and current document information.",
        "process": [
            "Select the renewal service.",
            "Provide or extract identity details from the current document.",
            "Confirm personal details and current address.",
            "Review the collected data and complete the request.",
        ],
        "questions": {
            "Full Name": "What is your full name as it should appear on the renewal request?",
            "Date of Birth": "What is your date of birth?",
            "Current ID Number": "What is your current ID number?",
            "Expiry Date": "What is the expiry date on your current ID?",
            "Address": "What is your current address?",
            "Blood Type": "What is your blood type?",
        },
    },
    "VISA Appointment": {
        "template": "visa_appointment.json",
        "description": "Book a visa appointment by collecting passport details, travel purpose, and a preferred appointment date.",
        "process": [
            "Choose the visa appointment service.",
            "Provide passport and nationality information.",
            "Specify destination country and travel purpose.",
            "Pick the desired appointment date and review the request.",
        ],
        "questions": {
            "Full Name": "What is your full name as it appears on your passport?",
            "Passport Number": "What is your passport number?",
            "Nationality": "What is your nationality?",
            "Destination Country": "Which country are you traveling to?",
            "Purpose of Travel": "What is the purpose of your travel?",
            "Desired Appointment Date": "What appointment date would you prefer?",
        },
    },
    "Driving License Renewal": {
        "template": "driving_license_renewal.json",
        "description": "Renew a driving license by confirming license details, vehicle class, and vision status.",
        "process": [
            "Select the driving license renewal service.",
            "Provide license details or upload the current license.",
            "Confirm vehicle class and issue date.",
            "Confirm vision test status and review the request.",
        ],
        "questions": {
            "Full Name": "What is your full name as it appears on your license?",
            "License Number": "What is your license number?",
            "Vehicle Class": "What vehicle class is on your license?",
            "Issue Date": "What is the issue date on your current license?",
            "Vision Test Status": "What is your current vision test status?",
        },
    },
}

SERVICE_ALIASES = {
    "id renewal": "ID Renewal",
    "id card renewal": "ID Renewal",
    "renew id": "ID Renewal",
    "visa appointment": "VISA Appointment",
    "visa": "VISA Appointment",
    "driving license renewal": "Driving License Renewal",
    "driver license renewal": "Driving License Renewal",
    "license renewal": "Driving License Renewal",
}

ASSISTANT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_services",
            "description": "List all available company services with short descriptions.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_service_details",
            "description": "Get the description, process, required fields, and current questions for a service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Canonical service name.",
                    }
                },
                "required": ["service_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "select_service",
            "description": "Select a service for the current session and create its submission record if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "description": "Canonical service name or close alias.",
                    }
                },
                "required": ["service_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_submission_state",
            "description": "Read the current submission database record for this session, including filled and missing fields.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_submission_fields",
            "description": "Update one or more form fields in the current submission database using grounded user or document data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "object",
                        "description": "Map of field name to extracted value.",
                        "additionalProperties": {"type": "string"},
                    }
                },
                "required": ["fields"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_service_request",
            "description": "Mark the current service request complete if no required fields are missing.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


@app.on_event("startup")
async def startup_event() -> None:
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def healthcheck() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": OLLAMA_MODEL,
        "sessions": get_session_count(),
        "services": list(SERVICE_CATALOG.keys()),
    }


@app.post("/chat", response_model=PromptResponse)
async def chat(request: Request) -> PromptResponse:
    session_id, prompt, reset, files = await parse_chat_request(request)
    session = get_or_create_session(session_id, reset=reset)

    if session.completed:
        return PromptResponse(
            session_id=session.session_id,
            response="This session is already complete. Start a new session if you need another service.",
            model=OLLAMA_MODEL,
            service_name=session.service_name,
            submission_path=session.submission_path,
            completed=True,
            missing_fields=[],
        )

    document_payload = await build_document_payload(files)
    user_content = build_user_content(prompt, document_payload)
    session_user_summary = build_session_user_summary(prompt, document_payload.filenames)
    update_session_state(
        session,
        message={"role": "user", "content": session_user_summary},
    )

    response_text = await run_assistant_turn(
        session=session,
        user_content=user_content,
    )

    form_data = load_current_form(session)
    missing_fields = get_missing_fields(form_data or {})

    return PromptResponse(
        session_id=session.session_id,
        response=response_text,
        model=OLLAMA_MODEL,
        service_name=session.service_name,
        submission_path=session.submission_path,
        completed=session.completed,
        missing_fields=missing_fields,
    )


@app.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
async def remove_session(session_id: str) -> DeleteSessionResponse:
    return DeleteSessionResponse(session_id=session_id, deleted=delete_session(session_id))


async def parse_chat_request(request: Request) -> tuple[str | None, str, bool, list[StarletteUploadFile]]:
    content_type = request.headers.get("content-type", "").lower()

    if "multipart/form-data" in content_type:
        form = await request.form()
        session_id = normalize_optional_string(form.get("session_id"))
        prompt = normalize_optional_string(form.get("prompt")) or ""
        reset = coerce_bool(form.get("reset"))
        files = [
            value
            for _, value in form.multi_items()
            if isinstance(value, StarletteUploadFile)
        ]
        return session_id, prompt, reset, files

    try:
        payload = PromptRequest.model_validate(await request.json())
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid chat payload.") from exc

    return payload.session_id, payload.prompt.strip(), payload.reset, []


async def run_assistant_turn(
    *,
    session: SessionState,
    user_content: list[dict[str, Any]],
) -> str:
    messages = build_llm_messages(session=session, user_content=user_content)

    for _ in range(MAX_TOOL_ROUNDS):
        completion = await ollama_chat_completion(
            messages,
            tools=ASSISTANT_TOOLS,
            temperature=0.1,
        )
        message = completion.get("message") or {}
        assistant_content = extract_message_content(message)
        assistant_tool_calls = list(message.get("tool_calls") or [])

        assistant_message_payload: dict[str, Any] = {
            "role": "assistant",
        }
        if assistant_content:
            assistant_message_payload["content"] = assistant_content
        if assistant_tool_calls:
            assistant_message_payload["tool_calls"] = assistant_tool_calls

        messages.append(assistant_message_payload)

        if not assistant_tool_calls:
            final_text = assistant_content.strip()
            if not final_text:
                raise HTTPException(status_code=502, detail="Ollama returned an empty response.")
            update_session_state(session, message={"role": "assistant", "content": final_text})
            return final_text

        for tool_call in assistant_tool_calls:
            tool_result = execute_tool_call(session=session, tool_call=tool_call)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": get_tool_call_name(tool_call),
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )
            update_session_state(
                session,
                message={
                    "role": "tool",
                    "content": summarize_tool_result(tool_result),
                },
            )

    raise HTTPException(status_code=502, detail="Ollama exceeded the maximum tool rounds.")


def build_llm_messages(session: SessionState, user_content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state_summary = build_state_summary(session)
    history_messages = build_history_messages(session)

    return [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n\n"
                f"{ASSISTANT_STYLE_PROMPT}\n\n"
                "Rules:\n"
                "- Use tools to inspect services and to read or modify submission state.\n"
                "- Use `select_service` before updating fields if no service is active.\n"
                "- Use `get_submission_state` before asking a new question when you need to verify what is still missing.\n"
                "- Use `update_submission_fields` to save grounded details from the user or uploaded documents.\n"
                "- If the user asks what a service includes, use `get_service_details` and explain it clearly.\n"
                "- When no service has been chosen yet, help the user choose by name and explain the options.\n"
                "- Do not mention internal tool names to the user.\n"
            ),
        },
        {
            "role": "system",
            "content": state_summary,
        },
        *history_messages,
        {
            "role": "user",
            "content": user_content,
        },
    ]


def build_state_summary(session: SessionState) -> str:
    form_data = load_current_form(session)
    missing_fields = get_missing_fields(form_data or {})
    service_name = session.service_name or "None"
    submission_path = session.submission_path or "None"
    form_json = json.dumps(form_data or {}, ensure_ascii=False)

    return (
        "Current session state:\n"
        f"- Service: {service_name}\n"
        f"- Submission path: {submission_path}\n"
        f"- Completed: {session.completed}\n"
        f"- Missing fields: {', '.join(missing_fields) if missing_fields else 'None'}\n"
        f"- Stored submission JSON: {form_json}"
    )


def build_history_messages(session: SessionState) -> list[dict[str, Any]]:
    raw_messages = session.messages
    if raw_messages and str(raw_messages[-1].get("role") or "").strip().lower() == "user":
        raw_messages = raw_messages[:-1]
    raw_messages = raw_messages[-RECENT_MESSAGE_COUNT:]
    formatted_messages: list[dict[str, Any]] = []

    for message in raw_messages:
        role = str(message.get("role") or "").strip().lower()
        content = normalize_optional_string(message.get("content")) or ""
        if not role or not content:
            continue
        if role not in {"user", "assistant"}:
            continue
        formatted_messages.append({"role": role, "content": content})

    return formatted_messages


def build_user_content(prompt: str, document_payload: Any) -> list[dict[str, Any]]:
    content_parts: list[dict[str, Any]] = []
    content_parts.append(
        {
            "type": "text",
            "text": prompt or "No text message was provided. Use the uploaded documents if relevant.",
        }
    )

    for text_block in document_payload.text_blocks:
        content_parts.append(
            {
                "type": "text",
                "text": f"Extracted PDF text:\n{text_block}",
            }
        )

    content_parts.extend(document_payload.vision_parts)
    return content_parts


def build_session_user_summary(prompt: str, filenames: list[str]) -> str:
    prompt_text = prompt or "No text message."
    if not filenames:
        return prompt_text
    return f"{prompt_text}\nUploaded files: {', '.join(filenames)}"


def execute_tool_call(session: SessionState, tool_call: Any) -> dict[str, Any]:
    function_payload = tool_call.get("function") if isinstance(tool_call, dict) else None
    if not isinstance(function_payload, dict):
        return {"ok": False, "error": "Malformed tool call payload."}

    function_name = str(function_payload.get("name") or "").strip()
    arguments = parse_tool_arguments(function_payload.get("arguments"))

    if function_name == "list_services":
        services = [
            {
                "name": service_name,
                "description": service_config["description"],
            }
            for service_name, service_config in SERVICE_CATALOG.items()
        ]
        return {"ok": True, "services": services}

    if function_name == "get_service_details":
        service_name = normalize_service_name(arguments.get("service_name"))
        if not service_name:
            return {"ok": False, "error": "Unknown service name."}
        return {
            "ok": True,
            "service": build_service_details(service_name),
        }

    if function_name == "select_service":
        service_name = normalize_service_name(arguments.get("service_name"))
        if not service_name:
            return {"ok": False, "error": "Unknown service name."}

        if session.service_name != service_name or not session.submission_path:
            submission_path = create_submission_from_template(session.session_id, service_name)
            update_session_state(
                session,
                service_name=service_name,
                submission_path=str(submission_path),
            )

        form_data = load_current_form(session) or {}
        return {
            "ok": True,
            "service_name": service_name,
            "submission_path": session.submission_path,
            "state": describe_submission_state(service_name, form_data),
        }

    if function_name == "get_submission_state":
        form_data = load_current_form(session)
        return {
            "ok": True,
            "service_name": session.service_name,
            "submission_path": session.submission_path,
            "state": describe_submission_state(session.service_name, form_data or {}),
        }

    if function_name == "update_submission_fields":
        if not session.service_name:
            return {"ok": False, "error": "No service selected yet."}

        form_data = load_current_form(session)
        if form_data is None:
            return {"ok": False, "error": "No submission database exists for this session."}

        fields = arguments.get("fields")
        if not isinstance(fields, dict):
            return {"ok": False, "error": "The fields argument must be an object."}

        applied_updates = apply_field_updates(form_data, fields)
        save_current_form(session, form_data)

        return {
            "ok": True,
            "service_name": session.service_name,
            "updated_fields": applied_updates,
            "state": describe_submission_state(session.service_name, form_data),
        }

    if function_name == "complete_service_request":
        form_data = load_current_form(session)
        missing_fields = get_missing_fields(form_data or {})
        if missing_fields:
            return {
                "ok": False,
                "error": "The service is not complete yet.",
                "missing_fields": missing_fields,
            }

        update_session_state(session, completed=True)
        return {
            "ok": True,
            "service_name": session.service_name,
            "submission_path": session.submission_path,
            "state": describe_submission_state(session.service_name, form_data or {}),
            "completed": True,
        }

    return {"ok": False, "error": f"Unknown tool: {function_name}"}


def build_service_details(service_name: str) -> dict[str, Any]:
    service_config = SERVICE_CATALOG[service_name]
    return {
        "name": service_name,
        "description": service_config["description"],
        "process": service_config["process"],
        "required_fields": list(service_config["questions"].keys()),
        "next_questions": service_config["questions"],
    }


def describe_submission_state(service_name: str | None, form_data: dict[str, Any]) -> dict[str, Any]:
    missing_fields = get_missing_fields(form_data)
    filled_fields = {
        key: value
        for key, value in form_data.items()
        if isinstance(value, str) and value.strip()
    }

    return {
        "service_name": service_name,
        "filled_fields": filled_fields,
        "missing_fields": missing_fields,
        "next_question": get_next_question(service_name, missing_fields),
        "is_complete": not missing_fields,
    }


def get_next_question(service_name: str | None, missing_fields: list[str]) -> str | None:
    if not service_name or not missing_fields:
        return None
    return SERVICE_CATALOG[service_name]["questions"].get(missing_fields[0])


def load_current_form(session: SessionState) -> dict[str, Any] | None:
    if not session.submission_path:
        return None

    submission_path = Path(session.submission_path)
    if not submission_path.exists():
        return None

    with submission_path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    if not isinstance(data, dict):
        raise HTTPException(status_code=500, detail="Stored submission file is invalid.")

    return data


def save_current_form(session: SessionState, form_data: dict[str, Any]) -> None:
    if not session.submission_path:
        raise HTTPException(status_code=500, detail="Submission path is missing for this session.")

    submission_path = Path(session.submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    with submission_path.open("w", encoding="utf-8") as file_handle:
        json.dump(form_data, file_handle, ensure_ascii=False, indent=2)


def create_submission_from_template(session_id: str, service_name: str) -> Path:
    service_config = SERVICE_CATALOG[service_name]
    template_path = TEMPLATES_DIR / str(service_config["template"])
    if not template_path.exists():
        raise HTTPException(status_code=500, detail=f"Template not found for {service_name}.")

    with template_path.open("r", encoding="utf-8") as file_handle:
        form_data = json.load(file_handle)

    safe_service_name = re.sub(r"[^a-z0-9]+", "_", service_name.lower()).strip("_")
    submission_path = SUBMISSIONS_DIR / f"{session_id}_{safe_service_name}.json"
    with submission_path.open("w", encoding="utf-8") as file_handle:
        json.dump(form_data, file_handle, ensure_ascii=False, indent=2)

    return submission_path


def apply_field_updates(form_data: dict[str, Any], field_updates: dict[str, Any]) -> list[str]:
    normalized_field_map = {normalize_key(key): key for key in form_data}
    applied_updates: list[str] = []

    for incoming_key, incoming_value in field_updates.items():
        canonical_key = normalized_field_map.get(normalize_key(str(incoming_key)))
        if not canonical_key:
            continue

        cleaned_value = normalize_optional_string(incoming_value)
        if cleaned_value is None:
            continue

        if form_data.get(canonical_key) == cleaned_value:
            continue

        form_data[canonical_key] = cleaned_value
        applied_updates.append(canonical_key)

    return applied_updates


def get_missing_fields(form_data: dict[str, Any]) -> list[str]:
    missing_fields: list[str] = []
    for key, value in form_data.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            missing_fields.append(key)
    return missing_fields


def extract_message_content(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            else:
                text_value = getattr(item, "text", None)
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def get_tool_call_name(tool_call: dict[str, Any]) -> str:
    function_payload = tool_call.get("function")
    if not isinstance(function_payload, dict):
        return ""
    return str(function_payload.get("name") or "").strip()


def summarize_tool_result(tool_result: dict[str, Any]) -> str:
    if tool_result.get("ok") is False:
        return f"Tool error: {tool_result.get('error')}"

    service_name = normalize_optional_string(tool_result.get("service_name"))
    state = tool_result.get("state")
    if isinstance(state, dict):
        missing_fields = state.get("missing_fields")
        updated_fields = tool_result.get("updated_fields")
        summary_parts = []
        if service_name:
            summary_parts.append(f"Service: {service_name}.")
        if isinstance(updated_fields, list) and updated_fields:
            summary_parts.append(f"Updated fields: {', '.join(str(item) for item in updated_fields)}.")
        if isinstance(missing_fields, list):
            summary_parts.append(
                "Missing fields: "
                + (", ".join(str(item) for item in missing_fields) if missing_fields else "none")
                + "."
            )
        return " ".join(summary_parts).strip() or json.dumps(tool_result, ensure_ascii=False)

    return json.dumps(tool_result, ensure_ascii=False)


def parse_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    if not isinstance(parsed, dict):
        return {}
    return parsed


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    return parse_json_object(str(raw_arguments or "{}"))


def normalize_service_name(value: Any) -> str | None:
    normalized = normalize_optional_string(value)
    if normalized is None:
        return None

    if normalized in SERVICE_CATALOG:
        return normalized

    return SERVICE_ALIASES.get(normalized.lower())


def normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    cleaned = str(value).strip()
    return cleaned or None


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
