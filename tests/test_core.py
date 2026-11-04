from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agrimind.api.app import Settings, create_app
from agrimind.corpus.ingest import IngestError, build_chunks, load_directory, parse_document
from agrimind.corpus.models import License
from agrimind.eval.benchmark import load_benchmark, run
from agrimind.gateway.routing import AskRequest, Mode, choose_mode, redact
from agrimind.kg.models import Certainty, Relation
from agrimind.kg.store import KnowledgeGraph
from agrimind.rag.answer import Assistant
from agrimind.rag.llm import StubLLM
from agrimind.rag.retriever import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "fixtures" / "corpus"


@pytest.fixture(scope="module")
def assistant() -> Assistant:
    return Assistant(BM25Retriever(build_chunks(load_directory(CORPUS))), StubLLM())


# --- kho tài liệu -----------------------------------------------------------

def test_non_commercial_excluded():
    docs = load_directory(CORPUS)
    assert any(d.license == License.CC_BY_NC for d in docs)
    chunks = build_chunks(docs)
    assert all(c.license.commercial_ok for c in chunks)
    assert not any(c.doc_id == "fx-khong-thuong-mai" for c in chunks)


def test_missing_license_rejected(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("---\nid: a\ntitle: t\nsource: s\n---\nnội dung", encoding="utf-8")
    with pytest.raises(IngestError):
        parse_document(p)


# --- đồ thị tri thức --------------------------------------------------------

def test_seed_graph_loads_and_excludes_correlation():
    g = KnowledgeGraph.from_yaml(ROOT / "data" / "kg" / "seed-tom-giam-an.yaml")
    causes = {r.subject for r in g.candidate_causes("out-giam-an")}
    assert "env-do-thap" in causes and "pa-wssv" in causes
    assert "out-mau-nuoc-dam" not in causes  # tương quan không phải nhân quả
    assert "MERGE" in g.to_cypher()


def test_high_certainty_needs_strong_evidence():
    with pytest.raises(ValueError):
        Relation(
            id="r", subject="a", type="causes", object="b",
            evidence=[{"source": "x", "grade": "D"}], certainty="high",
        )


def test_correlation_cannot_be_high():
    with pytest.raises(ValueError):
        Relation(
            id="r", subject="a", type="correlates_with", object="b",
            evidence=[{"source": "x", "grade": "A"}], certainty="high",
        )


# --- trợ lý -----------------------------------------------------------------

def test_answers_with_citation(assistant):
    ans = assistant.ask("Vì sao tôm giảm ăn vào sáng sớm, oxy hòa tan thấp?")
    assert not ans.refused
    assert ans.evidence and ans.evidence[0].source.startswith("fixture://")


def test_refuses_without_sources(assistant):
    ans = assistant.ask("Giá vàng hôm nay bao nhiêu?")
    assert ans.refused and ans.confidence == Certainty.INSUFFICIENT


def test_fabricated_citations_dropped(assistant):
    class Liar:
        def complete_json(self, system, user):
            return {"conclusion": "x", "evidence": [{"claim": "y", "source": "S99"}], "confidence": "high"}

    a = Assistant(assistant.retriever, Liar())
    ans = a.ask("oxy hòa tan ao tôm")
    assert ans.dropped_citations == 1
    assert ans.confidence == Certainty.INSUFFICIENT and ans.refused


# --- cổng API ---------------------------------------------------------------

def test_mode_selection():
    assert choose_mode(AskRequest(question="tôm giảm ăn")) == Mode.GENERAL
    assert choose_mode(AskRequest(question="ao số 3", farm_id="F123")) == Mode.FARM


def test_redact():
    out = redact(
        "Anh Ba, sđt 0912 345 678, email ba@x.vn, ao ở 10.123456, 105.654321, farm id FARM-0099",
        extra_terms=["Anh Ba"],
    )
    for leaked in ["Anh Ba", "0912", "ba@x.vn", "10.123456", "FARM-0099"]:
        assert leaked not in out, leaked


def _client(role: str) -> TestClient:
    s = Settings()
    s.role, s.corpus_dir, s.llm_url = role, CORPUS, ""
    return TestClient(create_app(s))


def test_core_rejects_farm_data():
    c = _client("core")
    r = c.post("/v1/ask", json={"question": "ao số 3 oxy thấp", "farm_id": "F1"})
    assert r.status_code == 403
    assert c.post("/v1/ask", json={"question": "oxy hòa tan ao tôm"}).status_code == 200


def test_country_endpoint_checks_country():
    c = _client("country")
    assert c.post("/v1/ask", json={"question": "oxy hòa tan ao tôm", "country": "TH"}).status_code == 421
    r = c.post("/v1/ask", json={"question": "oxy hòa tan ao tôm", "farm_id": "F1"})
    assert r.status_code == 200 and r.json()["evidence"]


# --- bộ chấm điểm -----------------------------------------------------------

def test_benchmark_runs(assistant):
    rep = run(assistant, load_benchmark(ROOT / "benchmarks" / "vi-core-v0.yaml"))
    s = rep.summary()
    assert s["items"] == 5
    assert s["no_fabricated_citations"] == 1.0
