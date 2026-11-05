"""Dịch vụ API AgriMind.

Cùng một mã chạy ở hai vai (A07), chọn bằng biến môi trường AGRIMIND_ROLE:
- `core`: AgriMind Core dùng chung toàn cầu. TỪ CHỐI mọi yêu cầu chế độ 2
  (có mã trang trại hoặc dữ liệu người dùng), để dữ liệu không bao giờ tới Core.
- `country`: điểm đặt trong từng nước, nhận cả hai chế độ.

Bảo vệ cho truy cập từ bên ngoài (xem security/clients.py):
- mọi đường /v1 cần khoá `Authorization: Bearer amk_...`, mỗi khoá có quyền riêng;
- giới hạn tần suất theo từng bên, giới hạn kích thước yêu cầu;
- nhật ký truy cập + dùng dữ liệu trại + dữ liệu đi ra Core (không ghi nội dung);
- môi trường production: không có tệp khoá thì KHÔNG khởi động, tắt trang /docs.

Biến môi trường:
  AGRIMIND_ENV          dev | production (mặc định dev)
  AGRIMIND_ROLE         core | country (mặc định country)
  AGRIMIND_COUNTRY      mã nước của điểm country, ví dụ VN
  AGRIMIND_CLIENTS_FILE tệp YAML các bên được truy cập (tạo khoá bằng `agrimind-keys new`)
  AGRIMIND_AUDIT_FILE   tệp nhật ký kiểm toán (JSON mỗi dòng)
  AGRIMIND_MAX_BODY     kích thước yêu cầu tối đa (byte), mặc định 65536
  AGRIMIND_CORPUS_DIR   thư mục tài liệu (.md có phần đầu YAML)
  AGRIMIND_KG_DIR       thư mục đồ thị tri thức (.yaml), mặc định data/kg
  AGRIMIND_TWIN_RULES   thư mục/tệp quy tắc cảnh báo, mặc định data/twin
  AGRIMIND_LAKE_DIR     (chỉ country) kho dữ liệu trại đã bỏ nhận dạng; trống = không nhận dữ liệu trại
  AGRIMIND_PSEUDONYM_KEY (chỉ country) khoá bí mật ≥ 32 ký tự để tạo mã giả cho trại
  AGRIMIND_LLM_URL      URL API tương thích OpenAI (vLLM chạy Qwen3); bỏ trống = StubLLM
  AGRIMIND_LLM_MODEL    tên mô hình, mặc định Qwen/Qwen3-8B
  AGRIMIND_QDRANT_URL   bật tìm theo nghĩa (gộp với BM25); cần thêm AGRIMIND_EMBED_URL
  AGRIMIND_EMBED_URL    API /embeddings tương thích OpenAI; AGRIMIND_EMBED_MODEL mặc định BAAI/bge-m3
  AGRIMIND_CORE_URL     (chỉ country) Core để hỏi khi kho trong nước chưa đủ; AGRIMIND_CORE_KEY là khoá
  NEXAGRI_TENANT_URL    (chỉ country) dịch vụ tenant của Nexagri để kiểm tra trại đã đồng ý chưa
  INTERNAL_SERVICE_KEY  (chỉ country) khoá gọi dịch vụ tenant của Nexagri
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from ..audit.log import AuditLog
from ..consent.gate import ConsentEvent, DenyAll, NexagriConsentGate
from ..corpus.ingest import build_chunks, load_directory
from ..gateway.egress import EgressStation
from ..gateway.routing import AskRequest, Mode, choose_mode, redact
from ..farmdata.lake import FarmDataLake, FarmRecord
from ..kg.store import KnowledgeGraph
from ..research.design import (PlotAssignment, randomize_blocks, sample_size_means,
                               sample_size_proportions)
from ..twin.rules import Alert, Reading, UnitState, evaluate, load_rules
from ..rag.answer import Answer, Assistant, PossibleCause
from ..rag.llm import OpenAICompatibleLLM, StubLLM
from ..rag.retriever import BM25Retriever
from ..rag.vector import HybridRetriever, OpenAICompatibleEmbedder, QdrantStore, VectorRetriever
from ..security.clients import ApiClient, ClientRegistry, Scope
from ..security.ratelimit import RateLimiter


class Settings:
    def __init__(self) -> None:
        self.env = os.getenv("AGRIMIND_ENV", "dev")
        self.role = os.getenv("AGRIMIND_ROLE", "country")
        if self.role not in {"core", "country"}:
            raise RuntimeError("AGRIMIND_ROLE phải là core hoặc country")
        self.country = os.getenv("AGRIMIND_COUNTRY", "VN").upper()
        self.clients_file = os.getenv("AGRIMIND_CLIENTS_FILE", "")
        self.audit_file = os.getenv("AGRIMIND_AUDIT_FILE", "")
        self.max_body = int(os.getenv("AGRIMIND_MAX_BODY", "65536"))
        self.corpus_dir = Path(os.getenv("AGRIMIND_CORPUS_DIR", "tests/fixtures/corpus"))
        self.kg_dir = os.getenv("AGRIMIND_KG_DIR", "data/kg")
        self.twin_rules = os.getenv("AGRIMIND_TWIN_RULES", "data/twin")
        self.lake_dir = os.getenv("AGRIMIND_LAKE_DIR", "")
        self.pseudonym_key = os.getenv("AGRIMIND_PSEUDONYM_KEY", "")
        self.llm_url = os.getenv("AGRIMIND_LLM_URL", "")
        self.llm_model = os.getenv("AGRIMIND_LLM_MODEL", "Qwen/Qwen3-8B")
        self.qdrant_url = os.getenv("AGRIMIND_QDRANT_URL", "")
        self.embed_url = os.getenv("AGRIMIND_EMBED_URL", "")
        self.embed_model = os.getenv("AGRIMIND_EMBED_MODEL", "BAAI/bge-m3")
        self.core_url = os.getenv("AGRIMIND_CORE_URL", "")
        self.core_key = os.getenv("AGRIMIND_CORE_KEY", "")
        self.nexagri_tenant_url = os.getenv("NEXAGRI_TENANT_URL", "")
        self.internal_key = os.getenv("INTERNAL_SERVICE_KEY", "")

    @property
    def production(self) -> bool:
        return self.env == "production"


@lru_cache
def get_graph(kg_dir: str) -> KnowledgeGraph | None:
    p = Path(kg_dir) if kg_dir else None
    return KnowledgeGraph.from_dir(p) if p and p.is_dir() else None


@lru_cache
def get_assistant(corpus_dir: str, llm_url: str, llm_model: str, qdrant_url: str = "",
                  embed_url: str = "", embed_model: str = "", kg_dir: str = "") -> Assistant:
    graph = get_graph(kg_dir)
    chunks = build_chunks(load_directory(Path(corpus_dir)))
    llm = OpenAICompatibleLLM(llm_url, llm_model) if llm_url else StubLLM()
    retriever = BM25Retriever(chunks)
    if qdrant_url and embed_url:
        vector = VectorRetriever(
            QdrantStore(qdrant_url, os.getenv("AGRIMIND_QDRANT_COLLECTION", "agrimind_corpus"),
                        os.getenv("AGRIMIND_QDRANT_API_KEY")),
            OpenAICompatibleEmbedder(embed_url, embed_model, os.getenv("AGRIMIND_EMBED_API_KEY")),
        )
        return Assistant(HybridRetriever(retriever, vector), llm, graph=graph)
    return Assistant(retriever, llm, graph=graph)


def make_consent_gate(s: Settings):
    if s.role != "country" or not s.nexagri_tenant_url or not s.internal_key:
        return DenyAll()
    return NexagriConsentGate(s.nexagri_tenant_url, s.internal_key)


# Bên "dev" chỉ dùng trên máy phát triển khi chưa có tệp khoá; không bao giờ có ở production.
DEV_CLIENT = ApiClient(id="dev", name="Máy phát triển", key_sha256="0" * 64,
                       scopes={Scope.ASK_GENERAL, Scope.ASK_FARM}, rate_per_minute=600)

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
}


# --- dạng yêu cầu/ trả lời của các API phụ ---------------------------------
class RecordBatch(BaseModel):
    records: list[FarmRecord] = Field(min_length=1, max_length=1000)


class TwinRequest(BaseModel):
    farm_id: str
    sector: str | None = None
    readings: list[Reading] = Field(min_length=1, max_length=5000)


class TwinResponse(BaseModel):
    states: list[UnitState]
    alerts: list[Alert]
    note: str


class SampleSizeRequest(BaseModel):
    kind: str = Field(pattern="^(means|proportions)$")
    delta: float | None = None
    sd: float | None = None
    p1: float | None = None
    p2: float | None = None
    alpha: float = 0.05
    power: float = 0.8


class DesignRequest(BaseModel):
    treatments: list[str] = Field(min_length=2, max_length=20)
    blocks: list[str] = Field(min_length=1, max_length=200)
    reps_per_block: int = Field(1, ge=1, le=20)
    seed: int = 0


def create_app(settings: Settings | None = None, consent_gate=None, registry: ClientRegistry | None = None,
               audit: AuditLog | None = None, egress: EgressStation | None = None,
               limiter: RateLimiter | None = None) -> FastAPI:
    s = settings or Settings()
    if registry is None and s.clients_file:
        registry = ClientRegistry.from_file(Path(s.clients_file))
    if registry is None and s.production:
        raise RuntimeError("production cần AGRIMIND_CLIENTS_FILE: không cho chạy API không khoá")
    audit = audit or AuditLog(Path(s.audit_file) if s.audit_file else None)
    limiter = limiter or RateLimiter()
    gate = consent_gate or make_consent_gate(s)
    if egress is None and s.role == "country" and s.core_url:
        egress = EgressStation(s.core_url, s.core_key, audit)

    lake = None
    if s.role == "country" and s.lake_dir:
        lake = FarmDataLake(Path(s.lake_dir), s.pseudonym_key)
    rules = load_rules(Path(s.twin_rules)) if s.twin_rules and Path(s.twin_rules).exists() else []

    docs = {} if not s.production else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title=f"AgriMind ({s.role})", version="0.3.0", **docs)
    app.state.consent_gate, app.state.audit = gate, audit

    @app.middleware("http")
    async def guard(request: Request, call_next):
        start = time.monotonic()
        if request.method in {"POST", "PUT", "PATCH"}:
            cl = request.headers.get("content-length")
            if cl is None:
                return JSONResponse({"detail": "Cần Content-Length."}, status_code=411, headers=_SECURITY_HEADERS)
            if not cl.isdigit() or int(cl) > s.max_body:
                return JSONResponse({"detail": "Yêu cầu quá lớn."}, status_code=413, headers=_SECURITY_HEADERS)
        response = await call_next(request)
        for k, v in _SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path.startswith("/v1/"):
            audit.write(
                "access", client=getattr(request.state, "client_id", None), method=request.method,
                path=request.url.path, status=response.status_code,
                ms=round((time.monotonic() - start) * 1000), ip=request.client.host if request.client else None,
            )
        return response

    def authenticate(request: Request) -> ApiClient:
        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth[:7].lower() == "bearer " else ""
        if registry is None:
            client = DEV_CLIENT  # chỉ tới được đây khi không phải production
        else:
            client = registry.authenticate(token)
            if client is None:
                raise HTTPException(401, "Thiếu hoặc sai khoá API.", headers={"WWW-Authenticate": "Bearer"})
        request.state.client_id = client.id
        wait = limiter.check(client.id, client.rate_per_minute)
        if wait:
            raise HTTPException(429, "Gọi quá nhiều, thử lại sau.", headers={"Retry-After": str(int(wait) + 1)})
        return client

    def need(client: ApiClient, scope: Scope) -> None:
        if scope not in client.scopes:
            raise HTTPException(403, f"Khoá này không có quyền {scope.value}.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "role": s.role, "country": s.country if s.role == "country" else None}

    @app.post("/v1/ask", response_model=Answer)
    def ask(req: AskRequest, client: ApiClient = Depends(authenticate)) -> Answer:
        mode = choose_mode(req)
        country = req.country.upper()
        if s.role == "core" and mode == Mode.FARM:
            raise HTTPException(403, "Core không nhận dữ liệu người dùng. Gửi câu hỏi này tới điểm API trong nước.")
        need(client, Scope.ASK_FARM if mode == Mode.FARM else Scope.ASK_GENERAL)
        if client.countries and country not in client.countries:
            raise HTTPException(403, f"Khoá này không được dùng cho nước {country}.")
        if s.role == "country" and country != s.country:
            raise HTTPException(421, f"Điểm API này phục vụ {s.country}, không phải {country}.")

        assistant = get_assistant(str(s.corpus_dir), s.llm_url, s.llm_model,
                                  s.qdrant_url, s.embed_url, s.embed_model, s.kg_dir)
        question = redact(req.question, req.private_terms) if mode == Mode.FARM else req.question
        # Dữ liệu trại chỉ được dùng khi trại đã bật chia sẻ trong Nexagri; không thì bỏ đi.
        farm_context = None
        if mode == Mode.FARM and req.attached_data and req.farm_id:
            if gate.allowed(req.farm_id):
                farm_context = req.attached_data
                audit.write("farm_data_used", client=client.id, tenant=req.farm_id, purpose="ask",
                            fields=sorted(map(str, req.attached_data))[:50])
            else:
                audit.write("farm_data_dropped", client=client.id, tenant=req.farm_id, reason="no_consent")
        ans = assistant.ask(question, domains=req.domains, farm_context=farm_context, lang=req.lang)
        # Kho trong nước chưa đủ → hỏi Core qua trạm kiểm soát (chỉ câu hỏi đã bỏ nhận dạng).
        if ans.refused and egress is not None:
            core = egress.ask_core(req.question, req.domains, req.private_terms, caller=client.id)
            if core is not None and not core.refused:
                return core
        return ans

    @app.get("/v1/kg/outcomes")
    def kg_outcomes(client: ApiClient = Depends(authenticate)) -> list[dict]:
        need(client, Scope.ASK_GENERAL)
        g = get_graph(s.kg_dir)
        if g is None:
            return []
        return [{"id": e.id, "name_vi": e.name_vi, "domains": e.domains}
                for e in g.entities.values() if e.type.value == "outcome"]

    @app.get("/v1/kg/causes/{outcome_id}", response_model=list[PossibleCause])
    def kg_causes(outcome_id: str, client: ApiClient = Depends(authenticate)) -> list[PossibleCause]:
        """Các yếu tố có quan hệ NHÂN QUẢ tới một kết quả (tương quan bị loại)."""
        need(client, Scope.ASK_GENERAL)
        g = get_graph(s.kg_dir)
        if g is None or outcome_id not in g.entities:
            raise HTTPException(404, "Không có kết quả này trong đồ thị.")
        out = g.entities[outcome_id]
        return [
            PossibleCause(outcome=out.name_vi, factor=g.entities[r.subject].name_vi, relation=r.type.value,
                          certainty=r.certainty, conditions=r.conditions, expert_approved=r.expert_approved)
            for r in g.candidate_causes(outcome_id)
        ]

    @app.post("/v1/internal/consent-events", status_code=204, response_class=Response)
    def consent_event(ev: ConsentEvent, client: ApiClient = Depends(authenticate)) -> Response:
        if s.role != "country":
            raise HTTPException(404)
        need(client, Scope.CONSENT_EVENTS)
        gate.apply(ev)
        audit.write("consent_event", client=client.id, tenant=ev.tenant_id, enabled=ev.enabled,
                    version=ev.consent_version)
        return Response(status_code=204)

    # --- dữ liệu trang trại (chỉ trong nước) ---------------------------------

    @app.post("/v1/farmdata/records")
    def farmdata_ingest(batch: RecordBatch, client: ApiClient = Depends(authenticate)) -> dict:
        if s.role != "country" or lake is None:
            raise HTTPException(404)
        need(client, Scope.FARMDATA_WRITE)
        allowed: dict[str, bool] = {}
        keep: list[FarmRecord] = []
        for r in batch.records:
            if r.tenant_id not in allowed:
                allowed[r.tenant_id] = gate.allowed(r.tenant_id)
            if allowed[r.tenant_id]:
                keep.append(r)
        written, flagged = lake.write(keep) if keep else (0, 0)
        for tid, ok in allowed.items():
            audit.write("farm_data_ingested" if ok else "farm_data_rejected", client=client.id, tenant=tid,
                        count=sum(r.tenant_id == tid for r in batch.records))
        return {"accepted": written, "flagged": flagged, "rejected_no_consent": len(batch.records) - written}

    @app.delete("/v1/farmdata/tenants/{tenant_id}")
    def farmdata_erase(tenant_id: str, client: ApiClient = Depends(authenticate)) -> dict:
        if s.role != "country" or lake is None:
            raise HTTPException(404)
        need(client, Scope.CONSENT_EVENTS)
        erased = lake.erase(tenant_id)
        audit.write("farm_data_erased", client=client.id, tenant=tenant_id, existed=erased)
        return {"erased": erased}

    # --- bản sao số mức 1–2: phản ánh + cảnh báo ---------------------------


    @app.post("/v1/twin/evaluate", response_model=TwinResponse)
    def twin_evaluate(req: TwinRequest, client: ApiClient = Depends(authenticate)) -> TwinResponse:
        if s.role != "country":
            raise HTTPException(403, "Dữ liệu trang trại chỉ xử lý ở điểm trong nước.")
        need(client, Scope.ASK_FARM)
        if not gate.allowed(req.farm_id):
            raise HTTPException(403, "Trại chưa bật chia sẻ dữ liệu cho AgriMind.")
        states, alerts = evaluate(req.readings, rules, req.sector)
        audit.write("farm_data_used", client=client.id, tenant=req.farm_id, purpose="twin",
                    fields=sorted({k for r in req.readings for k in r.values})[:50])
        return TwinResponse(states=states, alerts=alerts,
                            note="Chỉ là đề xuất; người phụ trách trại quyết định. Hệ thống không tự điều khiển thiết bị.")

    # --- phòng thí nghiệm AI: thiết kế thí nghiệm --------------------------

    @app.post("/v1/research/sample-size")
    def research_sample_size(req: SampleSizeRequest, client: ApiClient = Depends(authenticate)) -> dict:
        need(client, Scope.ASK_GENERAL)
        try:
            if req.kind == "means":
                n = sample_size_means(req.delta or 0, req.sd or 0, req.alpha, req.power)
            else:
                n = sample_size_proportions(req.p1 or 0, req.p2 or 0, req.alpha, req.power)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"per_group": n, "note": "Ước tính ban đầu; nhà thống kê duyệt trước khi làm thí nghiệm."}


    @app.post("/v1/research/design", response_model=list[PlotAssignment])
    def research_design(req: DesignRequest, client: ApiClient = Depends(authenticate)) -> list[PlotAssignment]:
        need(client, Scope.ASK_GENERAL)
        try:
            return randomize_blocks(req.treatments, req.blocks, req.reps_per_block, req.seed)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()
