"""Kiểm thử phần bổ sung giai đoạn 1: tìm theo nghĩa, nạp Neo4j, cổng đồng ý dữ liệu."""

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from agrimind.api.app import Settings, create_app
from agrimind.consent.gate import ConsentEvent, DenyAll, NexagriConsentGate
from agrimind.corpus.ingest import build_chunks, load_directory
from agrimind.kg import neo4j
from agrimind.kg.store import KnowledgeGraph
from agrimind.rag.answer import Assistant
from agrimind.rag.llm import StubLLM
from agrimind.rag.retriever import BM25Retriever
from agrimind.rag.vector import HybridRetriever, OpenAICompatibleEmbedder, QdrantStore, VectorRetriever

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "fixtures" / "corpus"
CHUNKS = build_chunks(load_directory(CORPUS))


# --- giả lập embedding + Qdrant --------------------------------------------

def _vec(text: str) -> list[float]:
    t = text.lower()
    return [float("oxy" in t or "tôm" in t), float("cà phê" in t or "đất" in t), 0.1]


class FakeQdrant:
    def __init__(self):
        self.points, self.created = {}, False

    def __call__(self, req: httpx.Request) -> httpx.Response:
        path, body = req.url.path, json.loads(req.content or b"{}")
        if req.url.host == "embed":
            return httpx.Response(200, json={"data": [
                {"index": i, "embedding": _vec(x)} for i, x in enumerate(body["input"])
            ]})
        if req.method == "GET":
            return httpx.Response(200 if self.created else 404, json={})
        if req.method == "PUT" and path.endswith("/points"):
            for p in body["points"]:
                self.points[p["id"]] = p
            return httpx.Response(200, json={"result": {}})
        if req.method == "PUT":
            self.created = True
            return httpx.Response(200, json={"result": True})
        if path.endswith("/points/search"):
            q = body["vector"]
            def cos(v):
                n = (sum(x * x for x in v) * sum(x * x for x in q)) ** 0.5
                return sum(a * b for a, b in zip(v, q)) / n if n else 0
            res = sorted(({"score": cos(p["vector"]), "payload": p["payload"]} for p in self.points.values()),
                         key=lambda r: -r["score"])[: body["limit"]]
            return httpx.Response(200, json={"result": res})
        return httpx.Response(400)


def _vector_retriever():
    fake = FakeQdrant()
    client = httpx.Client(transport=httpx.MockTransport(fake))
    vr = VectorRetriever(QdrantStore("http://qdrant", client=client),
                         OpenAICompatibleEmbedder("http://embed/v1", "bge-m3", client=client))
    return vr, fake


def test_vector_index_and_search_commercial_only():
    vr, fake = _vector_retriever()
    assert vr.index(CHUNKS) == len(CHUNKS)
    assert fake.created
    hits = vr.search("oxy trong ao tôm", k=3)
    assert hits and hits[0].relevant
    assert all(h.chunk.license.commercial_ok for h in hits)


def test_hybrid_finds_by_meaning_and_survives_outage():
    vr, _ = _vector_retriever()
    vr.index(CHUNKS)
    hybrid = HybridRetriever(BM25Retriever(CHUNKS), vr)
    # Câu hỏi không trùng từ khoá với BM25 nhưng cùng nghĩa (embedding giả coi "tôm" ~ oxy).
    ans = Assistant(hybrid, StubLLM()).ask("tôm")
    assert not ans.refused

    class Down(VectorRetriever):
        def search(self, *a, **k):
            raise httpx.ConnectError("qdrant down")

    bm25_only = HybridRetriever(BM25Retriever(CHUNKS), Down(vr.store, vr.embedder))
    assert not Assistant(bm25_only, StubLLM()).ask("oxy hòa tan ao tôm").refused
    assert Assistant(bm25_only, StubLLM()).ask("Giá vàng hôm nay bao nhiêu?").refused


# --- Neo4j ------------------------------------------------------------------

def test_neo4j_load_parameterised():
    g = KnowledgeGraph.from_yaml(ROOT / "data" / "kg" / "seed-tom-giam-an.yaml")
    sent = {}

    def handler(req):
        sent["body"] = json.loads(req.content)
        sent["path"] = req.url.path
        return httpx.Response(200, json={"results": [], "errors": []})

    n = neo4j.load(g, "http://neo4j:7474", "neo4j", "pw", client=httpx.Client(transport=httpx.MockTransport(handler)))
    stmts = sent["body"]["statements"]
    assert sent["path"] == "/db/neo4j/tx/commit"
    assert n == len(stmts) == 1 + len(g.entities) + len(g.relations)
    # tên tiếng Việt đi qua tham số, không nằm trong câu lệnh
    assert all("$" in s["statement"] for s in stmts[1:])
    corr = [s for s in stmts if "CORRELATES_WITH" in s["statement"]]
    assert corr and all(s["parameters"]["causal"] is False for s in corr)


def test_neo4j_errors_raised():
    g = KnowledgeGraph.from_yaml(ROOT / "data" / "kg" / "seed-tom-giam-an.yaml")
    bad = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={"errors": [{"code": "X", "message": "hỏng"}]})))
    try:
        neo4j.load(g, "http://neo4j:7474", "neo4j", "pw", client=bad)
    except neo4j.Neo4jError as e:
        assert "hỏng" in str(e)
    else:
        raise AssertionError("phải báo lỗi")


# --- cổng đồng ý dữ liệu ----------------------------------------------------

class FakeNexagri:
    def __init__(self):
        self.enabled = {"T-ON": True, "T-OFF": False}
        self.calls = 0
        self.down = False

    def __call__(self, req):
        self.calls += 1
        if self.down:
            raise httpx.ConnectError("down")
        if req.headers.get("X-Internal-Key") != "k":
            return httpx.Response(403)
        tid = req.url.path.split("/")[3]
        if tid not in self.enabled:
            return httpx.Response(404)
        return httpx.Response(200, json={"enabled": self.enabled[tid]})


def _gate(fake, clock=None):
    return NexagriConsentGate("http://tenant", "k", ttl=60, client=httpx.Client(transport=httpx.MockTransport(fake)),
                              clock=clock or (lambda: 0.0))


def test_gate_follows_nexagri_and_fails_closed():
    fake = FakeNexagri()
    g = _gate(fake)
    assert g.allowed("T-ON") and not g.allowed("T-OFF") and not g.allowed("T-NONE")
    assert not g.allowed("") and not g.allowed("../x")
    calls = fake.calls
    assert g.allowed("T-ON") and fake.calls == calls  # nhớ đệm
    fake.down = True
    assert not _gate(fake).allowed("T-ON")  # Nexagri không trả lời → coi như chưa đồng ý


def test_revoke_event_applies_immediately_and_old_events_ignored():
    fake = FakeNexagri()
    g = _gate(fake)
    assert g.allowed("T-ON")
    now = datetime(2026, 11, 5, tzinfo=timezone.utc)
    g.apply(ConsentEvent(tenant_id="T-ON", enabled=False, at=now))
    assert not g.allowed("T-ON")
    # sự kiện bật cũ hơn đến muộn: bỏ qua
    g.apply(ConsentEvent(tenant_id="T-ON", enabled=True, at=now.replace(hour=0, minute=0, second=0, day=4)))
    assert not g.allowed("T-ON")


def _client(gate, role="country"):
    s = Settings()
    s.role, s.corpus_dir, s.llm_url, s.qdrant_url = role, CORPUS, "", ""
    return TestClient(create_app(s, consent_gate=gate))


def test_farm_data_used_only_with_consent():
    gate = _gate(FakeNexagri())
    c = _client(gate)
    body = {"question": "oxy hòa tan ao tôm thấp", "attached_data": {"do_mg_l": 2.1}}
    r = c.post("/v1/ask", json={**body, "farm_id": "T-ON"})
    assert r.status_code == 200 and r.json()["farm_data_used"] is True
    r = c.post("/v1/ask", json={**body, "farm_id": "T-OFF"})
    assert r.status_code == 200 and r.json()["farm_data_used"] is False
    r = c.post("/v1/ask", json=body)  # không có mã trại → không xác minh được đồng ý
    assert r.json()["farm_data_used"] is False


def test_consent_event_endpoint():
    from agrimind.security.clients import ApiClient, ClientRegistry, Scope, new_key

    key, sha = new_key("nexagri-vn")
    pkey, psha = new_key("partner")
    reg = ClientRegistry([
        ApiClient(id="nexagri-vn", name="Nexagri", key_sha256=sha,
                  scopes={Scope.ASK_GENERAL, Scope.ASK_FARM, Scope.CONSENT_EVENTS}),
        ApiClient(id="partner", name="Đối tác", key_sha256=psha, scopes={Scope.ASK_GENERAL}),
    ])
    gate = _gate(FakeNexagri())
    s = Settings()
    s.corpus_dir, s.llm_url, s.qdrant_url = CORPUS, "", ""
    c = TestClient(create_app(s, consent_gate=gate, registry=reg))
    ev = {"tenant_id": "T-ON", "enabled": False, "at": "2026-11-05T00:00:00Z"}
    assert c.post("/v1/internal/consent-events", json=ev).status_code == 401
    assert c.post("/v1/internal/consent-events", json=ev,
                  headers={"Authorization": f"Bearer {pkey}"}).status_code == 403
    assert c.post("/v1/internal/consent-events", json=ev,
                  headers={"Authorization": f"Bearer {key}"}).status_code == 204
    assert not gate.allowed("T-ON")
    s2 = Settings()
    s2.role, s2.corpus_dir = "core", CORPUS
    core = TestClient(create_app(s2, consent_gate=DenyAll(), registry=reg))
    assert core.post("/v1/internal/consent-events", json=ev,
                     headers={"Authorization": f"Bearer {key}"}).status_code == 404
