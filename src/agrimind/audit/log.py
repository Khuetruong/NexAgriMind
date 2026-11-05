"""Nhật ký kiểm toán chỉ ghi thêm (JSON mỗi dòng).

Ghi: ai gọi, gọi gì, kết quả, khi nào; trại nào bị dùng dữ liệu; gì đã đi ra Core.
KHÔNG ghi nội dung câu hỏi hay dữ liệu trại (chỉ mã băm và độ dài), để nhật ký
không trở thành nơi rò rỉ dữ liệu.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class AuditLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict] = []  # bản trong bộ nhớ khi không có tệp (máy phát triển, kiểm thử)
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields) -> None:
        rec = {"at": datetime.now(timezone.utc).isoformat(), "kind": kind, **fields}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock:
            if self.path:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            else:
                self.records.append(rec)
                del self.records[:-10_000]
