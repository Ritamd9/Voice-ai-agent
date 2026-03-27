from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback keeps tests dependency-light
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


load_dotenv()

COLLECTION_NAME = "voice-rag-agent"
SUPPORTED_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
)


@dataclass(frozen=True, slots=True)
class AppConfig:
    openai_api_key: str = ""
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    collection_name: str = COLLECTION_NAME

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "AppConfig":
        return cls(
            openai_api_key=str(state.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")),
            qdrant_url=str(state.get("qdrant_url") or os.getenv("QDRANT_URL", "")),
            qdrant_api_key=str(state.get("qdrant_api_key") or os.getenv("QDRANT_API_KEY", "")),
        )

    @property
    def is_complete(self) -> bool:
        return all(
            [
                self.openai_api_key.strip(),
                self.qdrant_url.strip(),
                self.qdrant_api_key.strip(),
            ]
        )

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.openai_api_key.strip():
            missing.append("OPENAI_API_KEY")
        if not self.qdrant_url.strip():
            missing.append("QDRANT_URL")
        if not self.qdrant_api_key.strip():
            missing.append("QDRANT_API_KEY")
        return missing

