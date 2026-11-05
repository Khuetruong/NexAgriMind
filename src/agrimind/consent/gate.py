"""Cổng đồng ý dữ liệu: AgriMind trong nước chỉ dùng dữ liệu trại khi trại đã bật.

Nguồn sự thật là Nexagri (GET /v1/tenants/{id}/agrimind-consent, gọi bằng khoá
dịch vụ nội bộ). Quy tắc (A03, A07):
- Không hỏi được Nexagri → coi như CHƯA đồng ý (đóng khi lỗi).
- Kết quả được nhớ ngắn (mặc định 60 giây) để không gọi Nexagri mỗi câu hỏi.
- Nexagri có thể đẩy sự kiện bật/tắt tới `/v1/internal/consent-events`; sự kiện
  TẮT có hiệu lực ngay, kể cả khi bộ nhớ đệm còn hạn.
- Core không bao giờ có cổng này, vì Core không nhận dữ liệu trại.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable, Protocol

import httpx
from pydantic import BaseModel


class ConsentGate(Protocol):
    def allowed(self, tenant_id: str) -> bool: ...


class ConsentEvent(BaseModel):
    tenant_id: str
    enabled: bool
    consent_version: str | None = None
    at: datetime


class DenyAll:
    """Mặc định khi chưa cấu hình Nexagri: không dùng dữ liệu trại nào."""

    def allowed(self, tenant_id: str) -> bool:
        return False

    def apply(self, ev: ConsentEvent) -> None:
        pass


class NexagriConsentGate:
    def __init__(self, base_url: str, internal_key: str, ttl: float = 60, timeout: float = 5,
                 client: httpx.Client | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-Internal-Key": internal_key}
        self.ttl = ttl
        self.http = client or httpx.Client(timeout=timeout)
        self.clock = clock
        self._cache: dict[str, tuple[bool, float]] = {}
        self._last_event: dict[str, datetime] = {}
        self._lock = threading.Lock()

    def _fetch(self, tenant_id: str) -> bool:
        try:
            r = self.http.get(f"{self.base_url}/v1/tenants/{tenant_id}/agrimind-consent", headers=self.headers)
            if r.status_code != 200:
                return False
            return r.json().get("enabled") is True
        except (httpx.HTTPError, ValueError):
            return False

    def allowed(self, tenant_id: str) -> bool:
        tenant_id = (tenant_id or "").strip()
        if not tenant_id or "/" in tenant_id:
            return False
        now = self.clock()
        with self._lock:
            hit = self._cache.get(tenant_id)
            if hit and hit[1] > now:
                return hit[0]
        ok = self._fetch(tenant_id)
        with self._lock:
            self._cache[tenant_id] = (ok, now + self.ttl)
        return ok

    def apply(self, ev: ConsentEvent) -> None:
        """Nhận sự kiện từ Nexagri. Sự kiện cũ hơn sự kiện đã nhận thì bỏ qua."""
        with self._lock:
            last = self._last_event.get(ev.tenant_id)
            if last and ev.at < last:
                return
            self._last_event[ev.tenant_id] = ev.at
            if ev.enabled:
                # Bật: xoá bộ nhớ để lần sau hỏi lại Nexagri xác nhận, không tin sự kiện một mình.
                self._cache.pop(ev.tenant_id, None)
            else:
                self._cache[ev.tenant_id] = (False, self.clock() + self.ttl)
