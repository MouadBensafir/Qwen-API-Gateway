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
    SUBMISSIONS_DIR,
    SYSTEM_PROMPT,
    TEMPLATES_DIR,
    VLLM_MODEL,
)
from .document_utils import DocumentPayload, build_document_payload
from .models import DeleteSessionResponse, PromptRequest, PromptResponse
from .session_store import (
    SessionState,
    delete_session,
    get_or_create_session,
    get_session_count,
    update_session_state,
)
from .vllm_client import vllm_chat


app = FastAPI(title="IDWay Assist Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SERVICE_TEMPLATES = {
    "ID Renewal": "id_renewal.json",
    "VISA Appointment": "visa_appointment.json",
    "Driving License Renewal": "driving_license_renewal.json",
}

SERVICE_QUESTIONS = {
    "ID Renewal": {
        "Full Name": "What is your full name as it should appear on the renewal request?",
        "Date of Birth": "What is your date of birth?",
        "Current ID Number": "What is your current ID number?",
        "Expiry Date": "What is the expiry date on your current ID?",
        "Address": "What is your current address?",
        "Blood Type": "What is your blood type?",
    },
    "VISA Appointment": {
        "Full Name": "What is your full name as it appears on your passport?",
        "Passport Number": "What is your passport number?",
        "Nationality": "What is your nationality?",
        "Destination Country": "Which country are you traveling to?",
        "Purpose of Travel": "What is the purpose of your travel?",
        "Desired Appointment Date": "What appointment date would you prefer?",
    },
    "Driving License Renewal": {
        "Full Name": "What is your full name as it appears on your license?",
        "License Number": "What is your license number?",
        "Vehicle Class": "What vehicle class is on your license?",
        "Issue Date": "What is the issue date on your current license?",
        "Vision Test Status": "What is your current vision test status?",
    },
}

SERVICE_ALIASES = {
    "id renewal": "ID Renewal",
    "id card renewal": "ID Renewal",
    "renew id": "ID Renewal",
    "visa appointment": "VISA Appointment",
    "visa": "VISA Appointment",
    "driving license renewal": "Driving License Renewal",
    "license renewal": "Driving License Renewal",
    "driver license renewal": "Driving License Renewal",
}


@app.on_event("startup")
async def startup_event() -> None:
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def healthcheck() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": VLLM_MODEL,
        "sessions": get_session_count(),
        "services": list(SERVICE_TEMPLATES.keys()),
    }


@app.post("/chat", response_model=PromptResponse)
async def chat(request: Request) -> PromptResponse:
    session_id, prompt, reset, files = await parse_chat_request(request)
    session = get_or_create_session(session_id, reset=reset)

    if session.completed:
        return PromptResponse(
            session_id=session.session_id,
            response="This conversation session is already complete. Start a new session or reset the current one to request another service.",
            model=VLLM_MODEL,
            service_name=session.service_name,
            submission_path=session.submission_path,
            completed=True,
            missing_fields=[],
        )

    document_payload = await build_document_payload(files)
    current_form = load_current_form(session)
    extracted_turn = await extract_turn_data(
        session=session,
        prompt=prompt,
        current_form=current_form,
        document_payload=document_payload,
    )

    selected_service = session.service_name or extracted_turn.get("service_name") or detect_service_from_text(prompt)
    if selected_service and selected_service not in SERVICE_TEMPLATES:
        selected_service = None

    if not selected_service:
        response_text = build_service_selection_message()
        update_session_state(
            session,
            message={
                "role": "assistant",
                "content": response_text,
            },
        )
        return PromptResponse(
            session_id=session.session_id,
            response=response_text,
            model=VLLM_MODEL,
            missing_fields=[],
        )

    if session.service_name != selected_service or not session.submission_path:
        submission_path = create_submission_from_template(session.session_id, selected_service)
        update_session_state(
            session,
            service_name=selected_service,
            submission_path=str(submission_path),
        )

    form_data = load_current_form(session)
    applied_updates = apply_field_updates(form_data, extracted_turn.get("field_updates", {}))
    save_current_form(session, form_data)

    missing_fields = get_missing_fields(form_data)
    if not missing_fields:
        response_text = build_completion_message(selected_service, form_data, applied_updates)
        update_session_state(
            session,
            completed=True,
            message={"role": "assistant", "content": response_text},
        )
        return PromptResponse(
            session_id=session.session_id,
            response=response_text,
            model=VLLM_MODEL,
            service_name=selected_service,
            submission_path=session.submission_path,
            completed=True,
            missing_fields=[],
        )

    response_text = build_follow_up_message(
        service_name=selected_service,
        form_data=form_data,
        missing_fields=missing_fields,
        applied_updates=applied_updates,
        document_payload=document_payload,
    )
    update_session_state(
        session,
        message={"role": "assistant", "content": response_text},
    )

    return PromptResponse(
        session_id=session.session_id,
        response=response_text,
        model=VLLM_MODEL,
        service_name=selected_service,
        submission_path=session.submission_path,
        completed=False,
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


async def extract_turn_data(
    *,
    session: SessionState,
    prompt: str,
    current_form: dict[str, Any] | None,
    document_payload: DocumentPayload,
) -> dict[str, Any]:
    available_services = "\n".join(f"- {service}" for service in SERVICE_TEMPLATES)
    current_service = session.service_name or "None"
    current_form_json = json.dumps(current_form or {}, ensure_ascii=False, indent=2)

    content_parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"{SYSTEM_PROMPT}\n\n"
                "Extract service intent and grounded field values.\n"
                "Return only valid JSON with this shape:\n"
                '{'
                '"service_name": "ID Renewal|VISA Appointment|Driving License Renewal|null", '
                '"field_updates": {"Field Name": "value"}, '
                '"notes": "short note"'
                '}\n\n'
                f"Available services:\n{available_services}\n\n"
                f"Current selected service: {current_service}\n"
                f"Current form state:\n{current_form_json}\n\n"
                "Rules:\n"
                f"- Conversation style: {ASSISTANT_STYLE_PROMPT}\n"
                "- If a service is already selected, keep it unless the user clearly requests a different one.\n"
                "- Only populate fields that are explicitly stated in the user text or visible in uploaded documents.\n"
                "- Do not invent or normalize missing values beyond obvious whitespace cleanup.\n"
                "- If nothing is extractable for a field, omit it from field_updates.\n"
                f"User message:\n{prompt or '[no text provided]'}"
            ),
        }
    ]

    for text_block in document_payload.text_blocks:
        content_parts.append(
            {
                "type": "text",
                "text": f"Extracted PDF text:\n{text_block}",
            }
        )

    content_parts.extend(document_payload.vision_parts)

    raw_response = await vllm_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]
    )
    parsed = parse_json_object(raw_response)

    service_name = normalize_service_name(parsed.get("service_name"))
    field_updates = parsed.get("field_updates")
    if not isinstance(field_updates, dict):
        field_updates = {}

    return {
        "service_name": service_name,
        "field_updates": field_updates,
        "notes": str(parsed.get("notes") or "").strip(),
    }


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
    template_filename = SERVICE_TEMPLATES[service_name]
    template_path = TEMPLATES_DIR / template_filename
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


def build_service_selection_message() -> str:
    return (
        "Which service do you need?\n"
        "1. ID Renewal\n"
        "2. VISA Appointment\n"
        "3. Driving License Renewal"
    )


def build_follow_up_message(
    *,
    service_name: str,
    form_data: dict[str, Any],
    missing_fields: list[str],
    applied_updates: list[str],
    document_payload: DocumentPayload,
) -> str:
    next_field = missing_fields[0]
    question = SERVICE_QUESTIONS[service_name][next_field]

    preface_parts = [f"Service selected: {service_name}."]
    if document_payload.filenames:
        preface_parts.append(f"Processed document(s): {', '.join(document_payload.filenames)}.")
    if applied_updates:
        preface_parts.append(f"Updated: {', '.join(applied_updates)}.")

    remaining = ", ".join(missing_fields)
    preface_parts.append(f"Remaining fields: {remaining}.")
    preface_parts.append(question)
    return " ".join(preface_parts)


def build_completion_message(service_name: str, form_data: dict[str, Any], applied_updates: list[str]) -> str:
    lines = [f"{key}: {value}" for key, value in form_data.items()]
    response = [
        f"{service_name} is complete.",
    ]
    if applied_updates:
        response.append(f"Latest update: {', '.join(applied_updates)}.")
    response.append("Collected data:")
    response.extend(lines)
    response.append("This session is now closed.")
    return "\n".join(response)


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


def detect_service_from_text(prompt: str) -> str | None:
    lowered = prompt.lower()
    for alias, service_name in SERVICE_ALIASES.items():
        if alias in lowered:
            return service_name
    return None


def normalize_service_name(value: Any) -> str | None:
    normalized = normalize_optional_string(value)
    if normalized is None:
        return None

    if normalized in SERVICE_TEMPLATES:
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
