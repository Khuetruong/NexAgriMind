"""Kết nối mô hình ngôn ngữ.

Sản phẩm dùng Qwen3 chạy bằng vLLM (A04), vLLM có API tương thích OpenAI
nên chỉ cần một lớp client. `StubLLM` dùng cho kiểm thử và máy không có GPU.
"""

from __future__ import annotations

import json
from typing import Protocol

import httpx


class LLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = timeout

    def complete_json(self, system: str, user: str) -> dict:
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            timeout=self.timeout,
            json={
                "model": self.model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)


class StubLLM:
    """Trả lời cố định từ đoạn đầu tiên; chỉ để chạy thử luồng, không phải AI thật."""

    def complete_json(self, system: str, user: str) -> dict:
        ids = [line.split("]")[0].strip("[") for line in user.splitlines() if line.startswith("[S")]
        first = ids[0] if ids else None
        return {
            "conclusion": "Bản chạy thử: chưa có mô hình ngôn ngữ, chỉ trích các nguồn tìm được.",
            "evidence": [{"claim": "Xem nguồn được trích.", "source": first}] if first else [],
            "confidence": "low",
            "assumptions": ["Câu trả lời do StubLLM tạo, không phải kết luận khoa học."],
            "validation": ["Chuyên gia đọc các nguồn được trích."],
            "expert_check": ["Toàn bộ câu trả lời."],
        }
