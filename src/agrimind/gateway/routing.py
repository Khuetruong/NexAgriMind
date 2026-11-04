"""Cổng API: chọn chế độ và bỏ thông tin nhận dạng (A07).

Chế độ 1 (câu hỏi chung): không mã trang trại, không dữ liệu người dùng → lên Core.
Chế độ 2 (câu hỏi về trang trại): ở lại trong nước. Nếu cần kiến thức Core,
chỉ gửi câu hỏi đã qua `redact`.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field


class Mode(str, Enum):
    GENERAL = "general"  # chế độ 1
    FARM = "farm"  # chế độ 2


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    farm_id: str | None = None
    country: str = "VN"
    domains: list[str] | None = None
    attached_data: dict | None = None


def choose_mode(req: AskRequest) -> Mode:
    return Mode.FARM if (req.farm_id or req.attached_data) else Mode.GENERAL


_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.-]?\d){9,10}(?!\d)"), "[số điện thoại]"),
    # toạ độ thập phân: bỏ độ chính xác, giữ cấp vùng do lớp trên tự thêm
    (re.compile(r"-?\d{1,3}\.\d{3,}\s*,\s*-?\d{1,3}\.\d{3,}"), "[toạ độ]"),
    (re.compile(r"\b(?:farm|tt|trang trại)[\s_-]*(?:id)?[\s:#_-]*[A-Za-z0-9-]{4,}\b", re.I), "[trang trại]"),
]


def redact(text: str, extra_terms: list[str] | None = None) -> str:
    """Bỏ email, số điện thoại, toạ độ chính xác, mã trang trại và các tên được chỉ định.

    `extra_terms`: tên chủ trại, tên trang trại, địa chỉ lấy từ Nexagri – lớp gọi
    phải truyền vào, vì không thể đoán tên riêng chỉ bằng biểu thức chính quy.
    """
    out = text
    for term in sorted(extra_terms or [], key=len, reverse=True):
        if term.strip():
            out = re.sub(re.escape(term), "[ẩn]", out, flags=re.I)
    for pat, repl in _PATTERNS:
        out = pat.sub(repl, out)
    return out
