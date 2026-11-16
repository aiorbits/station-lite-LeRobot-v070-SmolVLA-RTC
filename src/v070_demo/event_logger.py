"""Structured artifact writer matching Section 12 deliverables."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DemoRunLogger:
    """Thread-safe writer for results/demo_three_object/<run_id>/ artifacts."""

    def __init__(self, output_dir: Path | None) -> None:
        self._lock = threading.RLock()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, filename: str, payload: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        with self._lock:
            target = self.output_dir / filename
            target.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8",
            )

    def append_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        if self.output_dir is None:
            return
        with self._lock:
            target = self.output_dir / filename
            line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            with target.open("a", encoding="utf-8") as fh:
                fh.write(line)

