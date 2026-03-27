from __future__ import annotations

from typing import Any, MutableMapping


DEFAULT_SESSION_STATE = {
    "openai_api_key": "",
    "qdrant_url": "",
    "qdrant_api_key": "",
    "selected_voice": "coral",
    "processed_documents": [],
    "client": None,
    "embedding_model": None,
    "processor_agent": None,
    "tts_agent": None,
}


def initialize_session_state(session_state: MutableMapping[str, Any]) -> None:
    for key, value in DEFAULT_SESSION_STATE.items():
        if key in session_state:
            continue
        session_state[key] = list(value) if isinstance(value, list) else value

