"""Khoá truy cập API cho các bên bên ngoài (Nexagri, đối tác, ứng dụng).

Quy tắc:
- Mỗi bên một khoá riêng, mỗi khoá chỉ được làm những việc ghi trong `scopes`.
- AgriMind chỉ lưu MÃ BĂM (SHA-256) của khoá, không lưu khoá. Lộ tệp cấu hình
  cũng không dùng được khoá.
- Khoá dạng `amk_<mã bên>.<chuỗi ngẫu nhiên 43 ký tự>`; tạo bằng `agrimind-keys new`.
- Khoá tạm khoá (`disabled: true`) hoặc hết hạn (`expires`) bị từ chối ngay.

Quyền (scope):
  ask:general     hỏi câu hỏi chung (không dữ liệu trại) – cấp được cho đối tác
  ask:farm        hỏi kèm mã trại / dữ liệu trại – chỉ cấp cho Nexagri trong nước
  consent:events  báo bật/tắt chia sẻ dữ liệu – chỉ cấp cho Nexagri trong nước
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

_KEY = re.compile(r"^amk_([a-z0-9][a-z0-9-]{1,62})\.([A-Za-z0-9_-]{32,128})$")


class Scope(str, Enum):
    ASK_GENERAL = "ask:general"
    ASK_FARM = "ask:farm"
    CONSENT_EVENTS = "consent:events"


class ApiClient(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,62}$")
    name: str
    key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scopes: set[Scope]
    countries: list[str] | None = None  # None = mọi nước mà điểm API phục vụ
    rate_per_minute: int = Field(default=60, ge=1, le=100_000)
    disabled: bool = False
    expires: datetime | None = None

    @field_validator("countries")
    @classmethod
    def _upper(cls, v: list[str] | None) -> list[str] | None:
        return [c.strip().upper() for c in v] if v else v

    def active(self, now: datetime | None = None) -> bool:
        if self.disabled:
            return False
        if self.expires is None:
            return True
        exp = self.expires if self.expires.tzinfo else self.expires.replace(tzinfo=timezone.utc)
        return (now or datetime.now(timezone.utc)) < exp


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key(client_id: str) -> tuple[str, str]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,62}", client_id):
        raise ValueError("mã bên chỉ gồm chữ thường, số, dấu gạch ngang (2–63 ký tự)")
    key = f"amk_{client_id}.{secrets.token_urlsafe(32)}"
    return key, hash_key(key)


class ClientRegistry:
    def __init__(self, clients: list[ApiClient]) -> None:
        self.clients: dict[str, ApiClient] = {}
        for c in clients:
            if c.id in self.clients:
                raise ValueError(f"trùng mã bên {c.id}")
            self.clients[c.id] = c

    @classmethod
    def from_file(cls, path: Path) -> "ClientRegistry":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls([ApiClient(**c) for c in data.get("clients", [])])

    def authenticate(self, token: str | None) -> ApiClient | None:
        m = _KEY.match((token or "").strip())
        if not m:
            return None
        client = self.clients.get(m.group(1))
        if client is None:
            # vẫn băm để thời gian trả lời không lộ mã bên nào tồn tại
            hmac.compare_digest(hash_key(token or ""), "0" * 64)
            return None
        if not hmac.compare_digest(hash_key(m.group(0)), client.key_sha256):
            return None
        return client if client.active() else None
