"""Bảo vệ API cho truy cập từ bên ngoài + trạm kiểm soát dữ liệu ra."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from agrimind.api.app import Settings, create_app
from agrimind.audit.log import AuditLog
from agrimind.consent.gate import DenyAll
from agrimind.gateway.egress import EgressStation, leaks
from agrimind.security.clients import ApiClient, ClientRegistry, Scope, hash_key, new_key
from agrimind.security.ratelimit import RateLimiter

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "fixtures" / "corpus"
Q = {"question": "oxy hòa tan ao tôm"}


class Allow:
    def allowed(self, tid):
        return True

    def apply(self, ev):
        pass


@pytest.fixture
def keys():
    out = {}
    clients = []
    for cid, scopes, extra in [
        ("nexagri-vn", {Scope.ASK_GENERAL, Scope.ASK_FARM, Scope.CONSENT_EVENTS}, {"countries": ["VN"]}),
        ("partner", {Scope.ASK_GENERAL}, {"rate_per_minute": 2}),
        ("off", {Scope.ASK_GENERAL}, {"disabled": True}),
        ("old", {Scope.ASK_GENERAL}, {"expires": datetime.now(timezone.utc) - timedelta(days=1)}),
    ]:
        k, sha = new_key(cid)
        out[cid] = k
        clients.append(ApiClient(id=cid, name=cid, key_sha256=sha, scopes=scopes, **extra))
    out["registry"] = ClientRegistry(clients)
    return out


def _app(keys, role="country", audit=None, gate=None, egress=None, **kw):
    s = Settings()
    s.role, s.corpus_dir, s.llm_url, s.qdrant_url = role, CORPUS, "", ""
    for k, v in kw.items():
        setattr(s, k, v)
    return TestClient(create_app(s, consent_gate=gate or DenyAll(), registry=keys["registry"],
                                 audit=audit or AuditLog(), egress=egress))


def H(k):
    return {"Authorization": f"Bearer {k}"}


def test_key_format_and_hash_only():
    k, sha = new_key("abc")
    assert k.startswith("amk_abc.") and len(k) > 40 and sha == hash_key(k) and k not in sha
    with pytest.raises(ValueError):
        new_key("Bad Id")


def test_auth_required_and_rejected(keys):
    c = _app(keys)
    assert c.get("/health").status_code == 200  # kiểm tra sống không cần khoá
    r = c.post("/v1/ask", json=Q)
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert c.post("/v1/ask", json=Q, headers=H("amk_partner.wrongwrongwrongwrongwrongwrongwrong")).status_code == 401
    assert c.post("/v1/ask", json=Q, headers=H(keys["off"])).status_code == 401
    assert c.post("/v1/ask", json=Q, headers=H(keys["old"])).status_code == 401
    assert c.post("/v1/ask", json=Q, headers=H(keys["partner"])).status_code == 200


def test_scopes_and_countries(keys):
    c = _app(keys)
    farm = {**Q, "farm_id": "T1"}
    assert c.post("/v1/ask", json=farm, headers=H(keys["partner"])).status_code == 403
    assert c.post("/v1/ask", json=farm, headers=H(keys["nexagri-vn"])).status_code == 200
    th = _app(keys, country="TH")
    assert th.post("/v1/ask", json={**Q, "country": "TH"}, headers=H(keys["nexagri-vn"])).status_code == 403


def test_rate_limit(keys):
    c = _app(keys)
    codes = [c.post("/v1/ask", json=Q, headers=H(keys["partner"])).status_code for _ in range(3)]
    assert codes[:2] == [200, 200] and codes[2] == 429


def test_body_limit_and_security_headers(keys):
    c = _app(keys, max_body=200)
    r = c.post("/v1/ask", json={"question": "x" * 500}, headers=H(keys["partner"]))
    assert r.status_code == 413
    r = c.post("/v1/ask", json=Q, headers=H(keys["partner"]))
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store"


def test_production_needs_keys_and_hides_docs(keys):
    s = Settings()
    s.env, s.corpus_dir, s.clients_file = "production", CORPUS, ""
    with pytest.raises(RuntimeError):
        create_app(s, consent_gate=DenyAll())
    prod = TestClient(create_app(s, consent_gate=DenyAll(), registry=keys["registry"]))
    assert prod.get("/docs").status_code == 404 and prod.get("/openapi.json").status_code == 404


def test_audit_has_no_content(keys):
    audit = AuditLog()
    c = _app(keys, audit=audit, gate=Allow())
    secret = "Anh Ba 0912345678"
    c.post("/v1/ask", json={"question": "oxy hòa tan ao tôm " + secret, "farm_id": "T1",
                            "attached_data": {"do_mg_l": 2.1, "owner": secret}}, headers=H(keys["nexagri-vn"]))
    kinds = [r["kind"] for r in audit.records]
    assert "access" in kinds and "farm_data_used" in kinds
    dumped = json.dumps(audit.records, ensure_ascii=False)
    assert "0912345678" not in dumped and "Anh Ba" not in dumped and "2.1" not in dumped


def test_registry_file(tmp_path, keys):
    k, sha = new_key("filed")
    p = tmp_path / "clients.yaml"
    p.write_text(f"clients:\n  - id: filed\n    name: F\n    key_sha256: {sha}\n    scopes: [ask:general]\n")
    assert ClientRegistry.from_file(p).authenticate(k).id == "filed"


def test_rate_limiter_refills():
    t = [0.0]
    rl = RateLimiter(clock=lambda: t[0])
    assert rl.check("a", 60) == 0
    for _ in range(59):
        rl.check("a", 60)
    assert rl.check("a", 60) > 0
    t[0] += 1.0
    assert rl.check("a", 60) == 0


# --- trạm kiểm soát dữ liệu ra ---------------------------------------------

def test_egress_sends_only_redacted_question():
    sent = []

    def core(req):
        sent.append((dict(req.headers), json.loads(req.content)))
        return httpx.Response(200, json={
            "question": "q", "conclusion": "Từ Core", "evidence": [{"claim": "c", "source_id": "x", "source": "s",
                                                                 "title": "t"}],
            "confidence": "medium"})

    audit = AuditLog()
    eg = EgressStation("http://core", "amk_vn.k", audit, client=httpx.Client(transport=httpx.MockTransport(core)))
    ans = eg.ask_core("Ao của Trại Hoà Bình bị giá vàng", ["D02", "bad"], ["Trại Hoà Bình"], caller="nexagri-vn")
    headers, body = sent[0]
    assert set(body) == {"question", "domains"} and body["domains"] == ["D02"]
    assert "Hoà Bình" not in body["question"] and headers["authorization"] == "Bearer amk_vn.k"
    assert ans.answered_by == "core" and ans.question.startswith("Ao của")
    assert [r["kind"] for r in audit.records] == ["egress"]


def test_egress_blocks_leaks_and_survives_core_down():
    assert leaks("gọi 0912 345 678") and leaks("a@b") and leaks("Trại X ở đây", ["trại x"])
    assert not leaks("oxy hòa tan thấp vào sáng sớm")
    audit = AuditLog()
    down = httpx.Client(transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x"))))
    eg = EgressStation("http://core", "", audit, client=down)
    assert eg.ask_core("số CMND 0791 2345 6789 01") is None  # dãy số dài còn sót → chặn
    assert eg.ask_core("giá vàng hôm nay") is None  # Core không trả lời → không sập
    assert [r["kind"] for r in audit.records] == ["egress_blocked", "egress", "egress_failed"]


def test_country_falls_back_to_core_when_local_refuses(keys):
    class FakeEgress:
        calls = []

        def ask_core(self, q, domains, terms, caller=""):
            from agrimind.kg.models import Certainty
            from agrimind.rag.answer import Answer
            self.calls.append(caller)
            return Answer(question=q, conclusion="Core", confidence=Certainty.LOW, answered_by="core",
                          evidence=[{"claim": "c", "source_id": "x", "source": "s", "title": "t"}])

    eg = FakeEgress()
    c = _app(keys, egress=eg)
    r = c.post("/v1/ask", json={"question": "Giá vàng hôm nay bao nhiêu?"}, headers=H(keys["partner"]))
    assert r.json()["answered_by"] == "core" and eg.calls == ["partner"]
    r = c.post("/v1/ask", json=Q, headers=H(keys["partner"]))
    assert r.json()["answered_by"] == "local" and len(eg.calls) == 1
