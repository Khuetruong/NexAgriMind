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
  AGRIMIND_QDRANT_URL  bật tìm theo nghĩa (gộp với BM25); cần thêm AGRIMIND_EMBED_URL
  AGRIMIND_EMBED_URL   API /embeddings tương thích OpenAI; AGRIMIND_EMBED_MODEL mặc định BAAI/bge-m3
  NEXAGRI_TENANT_URL   (chỉ country) dịch vụ tenant của Nexagri để kiểm tra trại đã đồng ý chưa
  INTERNAL_SERVICE_KEY (chỉ country) khoá gọi Nexagri và nhận sự kiện đồng ý từ Nexagri
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import hmac

from fastapi import FastAPI, Header, HTTPException, Response

from ..consent.gate import ConsentEvent, DenyAll, NexagriConsentGate
from ..corpus.ingest import build_chunks, load_directory
from ..gateway.routing import AskRequest, Mode, choose_mode, redact
from ..rag.answer import Answer, Assistant
from ..rag.llm import OpenAICompatibleLLM, StubLLM
from ..rag.retriever import BM25Retriever
from ..rag.vector import HybridRetriever, OpenAICompatibleEmbedder, QdrantStore, VectorRetriever


class Settings:
    def __init__(self) -> None:
        self.role = os.getenv("AGRIMIND_ROLE", "country")
        if self.role not in {"core", "country"}:
            raise RuntimeError("AGRIMIND_ROLE phải là core hoặc country")
        self.country = os.getenv("AGRIMIND_COUNTRY", "VN")
        self.corpus_dir = Path(os.getenv("AGRIMIND_CORPUS_DIR", "tests/fixtures/corpus"))
        self.llm_url = os.getenv("AGRIMIND_LLM_URL", "")
        self.llm_model = os.getenv("AGRIMIND_LLM_MODEL", "Qwen/Qwen3-8B")
        self.qdrant_url = os.getenv("AGRIMIND_QDRANT_URL", "")
        self.embed_url = os.getenv("AGRIMIND_EMBED_URL", "")
        self.embed_model = os.getenv("AGRIMIND_EMBED_MODEL", "BAAI/bge-m3")
        self.nexagri_tenant_url = os.getenv("NEXAGRI_TENANT_URL", "")
        self.internal_key = os.getenv("INTERNAL_SERVICE_KEY", "")


@lru_cache
def get_assistant(corpus_dir: str, llm_url: str, llm_model: str, qdrant_url: str = "",
                  embed_url: str = "", embed_model: str = "") -> Assistant:
    chunks = build_chunks(load_directory(Path(corpus_dir)))
    llm = OpenAICompatibleLLM(llm_url, llm_model) if llm_url else StubLLM()
    retriever = BM25Retriever(chunks)
    if qdrant_url and embed_url:
        vector = VectorRetriever(
            QdrantStore(qdrant_url, os.getenv("AGRIMIND_QDRANT_COLLECTION", "agrimind_corpus"),
                        os.getenv("AGRIMIND_QDRANT_API_KEY")),
            OpenAICompatibleEmbedder(embed_url, embed_model, os.getenv("AGRIMIND_EMBED_API_KEY")),
        )
        return Assistant(HybridRetriever(retriever, vector), llm)
    return Assistant(retriever, llm)


def make_consent_gate(s: Settings):
    if s.role != "country" or not s.nexagri_tenant_url or not s.internal_key:
        return DenyAll()
    return NexagriConsentGate(s.nexagri_tenant_url, s.internal_key)


def create_app(settings: Settings | None = None, consent_gate=None) -> FastAPI:
    s = settings or Settings()
    app = FastAPI(title=f"AgriMind ({s.role})", version="0.2.0")
    gate = consent_gate or make_consent_gate(s)
    app.state.consent_gate = gate

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
        assistant = get_assistant(str(s.corpus_dir), s.llm_url, s.llm_model,
                                  s.qdrant_url, s.embed_url, s.embed_model)
        question = redact(req.question) if mode == Mode.FARM else req.question
        # Dữ liệu trại chỉ được dùng khi trại đã bật chia sẻ trong Nexagri; không thì bỏ đi.
        farm_context = None
        if mode == Mode.FARM and req.attached_data and req.farm_id and gate.allowed(req.farm_id):
            farm_context = req.attached_data
        return assistant.ask(question, domains=req.domains, farm_context=farm_context)

    @app.post("/v1/internal/consent-events", status_code=204, response_class=Response)
    def consent_event(ev: ConsentEvent, x_internal_key: str = Header(default="")) -> Response:
        if s.role != "country":
            raise HTTPException(status_code=404)
        if not s.internal_key or not hmac.compare_digest(x_internal_key.encode(), s.internal_key.encode()):
            raise HTTPException(status_code=401, detail="Sai khoá dịch vụ nội bộ.")
        gate.apply(ev)
        return Response(status_code=204)

    return app


app = create_app()
