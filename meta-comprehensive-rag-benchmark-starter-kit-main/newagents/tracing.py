from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TraceWriter:
    """Append-only JSONL tracing that never records keys or image payloads."""

    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def write(self, event: str, **payload: Any) -> None:
        if self.path is None:
            return
        safe: Dict[str, Any] = {"event": event, "timestamp": round(time.time(), 3)}
        for key, value in payload.items():
            lowered = key.lower()
            if "key" in lowered or "base64" in lowered or lowered in {"image", "image_data"}:
                continue
            safe[key] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(safe, ensure_ascii=False, default=str)
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass

