"""MemPalace archival wrapper with silent fallback.

If mempalace is installed, each call to `store` upserts a drawer into the
local palace at ~/.mempalace/superfermion-benchmark. If not, the wrapper
silently no-ops — the harness still writes industrial_benchmark_results.json.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

PALACE_PATH = os.path.expanduser("~/.mempalace/superfermion-benchmark")


class MemoryStore:
    def __init__(self, palace_path: str = PALACE_PATH, enabled: bool = True):
        self.enabled = enabled
        self.palace_path = palace_path
        self._collection = None
        self._last_error: Optional[str] = None
        if enabled:
            try:
                os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
                os.makedirs(palace_path, exist_ok=True)
                from mempalace.palace import get_collection  # type: ignore
                self._collection = get_collection(
                    palace_path, collection_name="mempalace_drawers", create=True
                )
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                self.enabled = False

    @property
    def status(self) -> str:
        if self.enabled and self._collection is not None:
            try:
                return f"ok (count={self._collection.count()})"
            except Exception as e:
                return f"degraded ({type(e).__name__}: {e})"
        return f"disabled ({self._last_error or 'mempalace unavailable'})"

    def store(
        self,
        *,
        wing: str,
        room: str,
        content: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Upsert one drawer. Returns True on success, False on any error."""
        if not self.enabled or self._collection is None:
            return False
        try:
            drawer_id_src = f"{wing}|{room}|{source}|{hashlib.sha256(content.encode()).hexdigest()[:12]}"
            drawer_id = "drawer_" + hashlib.sha256(drawer_id_src.encode()).hexdigest()[:32]
            md: Dict[str, Any] = {
                "wing": wing,
                "room": room,
                "source_file": source,
                "added_by": "superfermion-industrial-benchmark",
                "filed_at": datetime.utcnow().isoformat() + "Z",
            }
            if metadata:
                # Chroma metadata must be str/int/float/bool — stringify anything else
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        md[k] = v
                    else:
                        md[k] = json.dumps(v, default=str)
            self._collection.upsert(
                documents=[content],
                ids=[drawer_id],
                metadatas=[md],
            )
            return True
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            return False

    def store_json(self, *, wing: str, room: str, payload: Dict[str, Any], source: str) -> bool:
        """Convenience: pretty-print a JSON blob as a drawer."""
        content = json.dumps(payload, indent=2, default=str)
        return self.store(wing=wing, room=room, content=content, source=source, metadata=payload)
