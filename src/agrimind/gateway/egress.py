"""Trạm kiểm soát dữ liệu ra (A07): lối DUY NHẤT từ điểm trong nước lên Core.

Chỉ gửi đi: câu hỏi đã bỏ thông tin nhận dạng + mã lĩnh vực. Không gửi mã trại,
dữ liệu trại, tên nước hay bất kỳ trường nào khác. Kiểm tra lại sau khi bỏ thông
tin; còn sót thứ gì giống email/số điện thoại/toạ độ thì CHẶN, không gửi.
Mỗi lần gửi (hoặc chặn) đều ghi nhật ký bằng mã băm, không ghi nội dung.
"""

from __future__ import annotations

import re

import httpx

from ..audit.log import AuditLog, digest
from ..rag.answer import Answer
from .routing import redact

_LEAK = [
    re.compile(r"@"),
    re.compile(r"\d[\d\s.-]{7,}\d"),  # dãy số dài: điện thoại, số giấy tờ, toạ độ
]


def leaks(text: str, private_terms: list[str] | None = None) -> bool:
    low = text.lower()
    if any(t.strip() and t.strip().lower() in low for t in private_terms or []):
        return True
    return any(p.search(text) for p in _LEAK)


class EgressStation:
    def __init__(self, core_url: str, api_key: str, audit: AuditLog, timeout: float = 60,
                 client: httpx.Client | None = None) -> None:
        self.core_url = core_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.audit = audit
        self.http = client or httpx.Client(timeout=timeout)

    def ask_core(self, question: str, domains: list[str] | None = None,
                 private_terms: list[str] | None = None, caller: str = "") -> Answer | None:
        text = redact(question, private_terms)
        if leaks(text, private_terms):
            self.audit.write("egress_blocked", caller=caller, sha=digest(text), length=len(text))
            return None
        payload: dict = {"question": text}
        if domains:
            payload["domains"] = [d for d in domains if re.fullmatch(r"D\d{2}", d)]
        self.audit.write("egress", caller=caller, target="core", sha=digest(text), length=len(text),
                         redacted=text != question)
        try:
            r = self.http.post(f"{self.core_url}/v1/ask", json=payload, headers=self.headers)
            r.raise_for_status()
            ans = Answer.model_validate(r.json())
        except (httpx.HTTPError, ValueError):
            self.audit.write("egress_failed", caller=caller, sha=digest(text))
            return None
        ans.answered_by = "core"
        ans.question = question  # trả lại câu gốc cho người hỏi trong nước
        return ans
