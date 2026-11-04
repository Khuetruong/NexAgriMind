"""Dịch vụ API AgriMind.

Cùng một mã chạy ở hai vai (A07), chọn bằng biến môi trường AGRIMIND_ROLE:
- `core`: AgriMind Core dùng chung toàn cầu. TỪ CHỐI mọi yêu cầu chế độ 2
  (có mã trang trại hoặc dữ liệu người dùng), để dữ liệu không bao giờ tới Core.
- `country`: điểm đặt trong từng nước, nhận cả hai chế độ.

Biến môi trường:
  AGRIMIND_ROLE        core | country (mặc định country)
  AGRIMIND_COUNTRY     mã nước của điểm country, ví dụ VN
  AGRIMIND_CORPUS_DIR  thư mục tài liệu (.md có phần đầu YAML)
  AGRIMIND_LLM_URL     URL API tương thích OpenAI (vLLM chạy Qwen3); bỏ trống = StubLLM
  AGRIMIND_LLM_MODEL   tên mô hình, mặc định Qwen/Qwen3-8B
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from ..corpus.ingest import build_chunks, load_directory
from ..gateway.routing import AskRequest, Mode, choose_mode, redact
from ..rag.answer import Answer, Assistant
from ..rag.llm import OpenAICompatibleLLM, StubLLM
from ..rag.retriever import BM25Retriever


class Settings:
    def __init__(self) -> None:
        self.role = os.getenv("AGRIMIND_ROLE", "country")
        if self.role not in {"core", "country"}:
            raise RuntimeError("AGRIMIND_ROLE phải là core hoặc country")
        self.country = os.getenv("AGRIMIND_COUNTRY", "VN")
        self.corpus_dir = Path(os.getenv("AGRIMIND_CORPUS_DIR", "tests/fixtures/corpus"))
        self.llm_url = os.getenv("AGRIMIND_LLM_URL", "")
        self.llm_model = os.getenv("AGRIMIND_LLM_MODEL", "Qwen/Qwen3-8B")


@lru_cache
def get_assistant(corpus_dir: str, llm_url: str, llm_model: str) -> Assistant:
    chunks = build_chunks(load_directory(Path(corpus_dir)))
    llm = OpenAICompatibleLLM(llm_url, llm_model) if llm_url else StubLLM()
    return Assistant(BM25Retriever(chunks), llm)


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()
    app = FastAPI(title=f"AgriMind ({s.role})", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "role": s.role, "country": s.country if s.role == "country" else None}

    @app.post("/v1/ask", response_model=Answer)
    def ask(req: AskRequest) -> Answer:
        mode = choose_mode(req)
        if s.role == "core" and mode == Mode.FARM:
            raise HTTPException(
                status_code=403,
                detail="Core không nhận dữ liệu người dùng. Gửi câu hỏi này tới điểm API trong nước.",
            )
        if s.role == "country" and req.country != s.country:
            raise HTTPException(status_code=421, detail=f"Điểm API này phục vụ {s.country}, không phải {req.country}.")
        assistant = get_assistant(str(s.corpus_dir), s.llm_url, s.llm_model)
        question = redact(req.question) if mode == Mode.FARM else req.question
        return assistant.ask(question, domains=req.domains)

    return app


app = create_app()
