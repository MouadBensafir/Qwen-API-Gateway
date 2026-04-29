from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass
class SessionState:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    service_name: str | None = None
    submission_path: str | None = None
    completed: bool = False


_SESSIONS: dict[str, SessionState] = {}
_LOCK = Lock()


def get_or_create_session(session_id: str | None, reset: bool = False) -> SessionState:
    with _LOCK:
        if session_id and not reset and session_id in _SESSIONS:
            return _SESSIONS[session_id]

        next_session_id = session_id or uuid4().hex
        state = SessionState(session_id=next_session_id)
        _SESSIONS[next_session_id] = state
        return state


def update_session_state(
    session: SessionState,
    *,
    service_name: str | None = None,
    submission_path: str | None = None,
    completed: bool | None = None,
    message: dict[str, Any] | None = None,
) -> None:
    with _LOCK:
        if service_name is not None:
            session.service_name = service_name
        if submission_path is not None:
            session.submission_path = submission_path
        if completed is not None:
            session.completed = completed
        if message is not None:
            session.messages.append(message)


def get_session_count() -> int:
    with _LOCK:
        return len(_SESSIONS)


def delete_session(session_id: str) -> bool:
    with _LOCK:
        return _SESSIONS.pop(session_id, None) is not None
