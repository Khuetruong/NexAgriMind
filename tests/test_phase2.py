"""Khung giai đoạn sau: kho dữ liệu trại, bản sao số, thiết kế thí nghiệm, tiếng Anh."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agrimind.api.app import Settings, create_app
from agrimind.audit.log import AuditLog
from agrimind.farmdata.lake import FarmDataLake, FarmRecord
from agrimind.research.design import (permutation_test, randomize_blocks, sample_size_means,
                                      sample_size_proportions)
from agrimind.security.clients import ApiClient, ClientRegistry, Scope, new_key
from agrimind.twin.rules import Reading, evaluate, load_rules

ROOT = Path(__file__).resolve().parents[1]
KEY = "k" * 40
NOW = datetime(2026, 11, 5, 6, 0, tzinfo=timezone.utc)


class Gate:
    def __init__(self, on):
        self.on = set(on)

    def allowed(self, t):
        return t in self.on

    def apply(self, ev):
        pass


# --- kho dữ liệu trại --------------------------------------------------------

def _rec(tid="T-ON", **kw):
    base = dict(tenant_id=tid, kind="telemetry", at=NOW, plot_id="AO-3", sector="aquaculture",
                lat=10.123456, lon=105.654321, values={"do": 4.2, "ph": 7.9},
                note="Anh Ba gọi 0912 345 678 báo tôm nổi đầu")
    base.update(kw)
    return FarmRecord(**base)


def test_lake_deidentifies_and_checks(tmp_path):
    lake = FarmDataLake(tmp_path, KEY)
    assert lake.write([_rec(), _rec(values={"ph": 19.0}, at=NOW + timedelta(days=3))]) == (2, 1)
    files = list(tmp_path.rglob("*.jsonl"))
    raw = "".join(f.read_text(encoding="utf-8") for f in files)
    for leaked in ["T-ON", "AO-3", "0912", "10.123456", "105.654321"]:
        assert leaked not in raw, leaked
    recs = lake.read("T-ON")
    assert recs[0].lat == 10.1 and recs[0].quality.value == "checked"
    assert recs[1].quality.value == "raw" and any("ph=19.0" in f for f in recs[1].flags)
    assert lake.pseudonym("T-ON") != FarmDataLake(tmp_path / "x", "z" * 40).pseudonym("T-ON")
    assert lake.erase("T-ON") and lake.read("T-ON") == [] and not lake.erase("T-ON")
    with pytest.raises(ValueError):
        FarmDataLake(tmp_path, "ngan")


# --- API ---------------------------------------------------------------------

@pytest.fixture
def api(tmp_path):
    keys, clients = {}, []
    for cid, scopes in [
        ("nexagri-vn", {Scope.ASK_GENERAL, Scope.ASK_FARM, Scope.CONSENT_EVENTS, Scope.FARMDATA_WRITE}),
        ("partner", {Scope.ASK_GENERAL}),
    ]:
        k, sha = new_key(cid)
        keys[cid] = {"Authorization": f"Bearer {k}"}
        clients.append(ApiClient(id=cid, name=cid, key_sha256=sha, scopes=scopes, rate_per_minute=1000))
    s = Settings()
    s.corpus_dir, s.llm_url, s.qdrant_url = ROOT / "tests/fixtures/corpus", "", ""
    s.lake_dir, s.pseudonym_key = str(tmp_path / "lake"), KEY
    s.twin_rules, s.kg_dir = str(ROOT / "data/twin"), str(ROOT / "data/kg")
    audit = AuditLog()
    c = TestClient(create_app(s, consent_gate=Gate({"T-ON"}), registry=ClientRegistry(clients), audit=audit))
    return c, keys, audit, tmp_path / "lake"


def test_farmdata_api_consent_and_erase(api):
    c, keys, audit, lake_dir = api
    recs = [_rec().model_dump(mode="json"), _rec("T-OFF").model_dump(mode="json")]
    assert c.post("/v1/farmdata/records", json={"records": recs}, headers=keys["partner"]).status_code == 403
    r = c.post("/v1/farmdata/records", json={"records": recs}, headers=keys["nexagri-vn"]).json()
    assert r == {"accepted": 1, "flagged": 0, "rejected_no_consent": 1}
    kinds = [x["kind"] for x in audit.records]
    assert "farm_data_ingested" in kinds and "farm_data_rejected" in kinds
    assert c.delete("/v1/farmdata/tenants/T-ON", headers=keys["nexagri-vn"]).json() == {"erased": True}
    assert not list(lake_dir.rglob("*.jsonl"))


def test_twin_alerts_suggest_only(api):
    c, keys, _, _ = api
    readings = [
        {"unit_id": "ao-1", "at": (NOW + timedelta(hours=i)).isoformat(), "values": {"do": v, "ph": p}}
        for i, (v, p) in enumerate([(4.5, 8.0), (2.8, 8.2), (2.5, 9.8)])
    ]
    body = {"farm_id": "T-ON", "sector": "aquaculture", "readings": readings}
    r = c.post("/v1/twin/evaluate", json=body, headers=keys["nexagri-vn"])
    assert r.status_code == 200
    out = r.json()
    ids = [a["rule_id"] for a in out["alerts"]]
    assert ids[0] == "aq-ph-bat-thuong" and "aq-do-thap" in ids  # mức 3 xếp trước
    assert all(a["action"] == "suggest_only" for a in out["alerts"])
    assert out["alerts"][0]["requires_expert_confirmation"] is True
    assert "ngưỡng nháp" in out["alerts"][0]["message"]
    assert out["states"][0]["trend"]["do"] == -2.0
    assert c.post("/v1/twin/evaluate", json={**body, "farm_id": "T-OFF"}, headers=keys["nexagri-vn"]).status_code == 403
    # quy tắc thủy sản không áp cho trại trồng trọt
    r = c.post("/v1/twin/evaluate", json={**body, "sector": "crop"}, headers=keys["nexagri-vn"]).json()
    assert r["alerts"] == []


def test_research_api(api):
    c, keys, _, _ = api
    r = c.post("/v1/research/sample-size", json={"kind": "means", "delta": 0.5, "sd": 1.0},
               headers=keys["partner"])
    assert r.json()["per_group"] == 63
    assert c.post("/v1/research/sample-size", json={"kind": "proportions", "p1": 0.5, "p2": 0.5},
                  headers=keys["partner"]).status_code == 400
    plots = c.post("/v1/research/design", json={"treatments": ["doi-chung", "vi-sinh"], "blocks": ["ao-1", "ao-2"],
                                                 "reps_per_block": 2, "seed": 7}, headers=keys["partner"]).json()
    assert len(plots) == 8 and sum(p["treatment"] == "vi-sinh" for p in plots if p["block"] == "ao-1") == 2


def test_english_answers(api):
    c, keys, _, _ = api
    r = c.post("/v1/ask", json={"question": "oxy hòa tan ao tôm", "lang": "en"}, headers=keys["partner"]).json()
    assert r["conclusion"].startswith("Test run") and not r["refused"]
    r = c.post("/v1/ask", json={"question": "price of gold today", "lang": "en"}, headers=keys["partner"]).json()
    assert r["refused"] and r["conclusion"].startswith("Not enough evidence")
    assert c.post("/v1/ask", json={"question": "abc def", "lang": "fr"}, headers=keys["partner"]).status_code == 422


# --- thống kê ----------------------------------------------------------------

def test_statistics():
    assert sample_size_means(0.5, 1.0) == 63
    assert sample_size_proportions(0.8, 0.9) in range(195, 205)
    a = randomize_blocks(["A", "B", "C"], ["k1", "k2"], seed=1)
    assert a == randomize_blocks(["A", "B", "C"], ["k1", "k2"], seed=1)  # tái lập được
    assert sorted(x.treatment for x in a if x.block == "k1") == ["A", "B", "C"]
    same = permutation_test([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], permutations=2000)
    diff = permutation_test([1, 2, 3, 4, 5], [11, 12, 13, 14, 15], permutations=2000)
    assert same.p_value > 0.5 and diff.p_value < 0.05 and diff.difference == 10


def test_draft_rules_load():
    rules = load_rules(ROOT / "data/twin")
    assert {r.id for r in rules} >= {"aq-do-thap", "po-nong-chuong", "cr-dat-kho"}
    assert not any(r.expert_approved for r in rules)
    _, alerts = evaluate([Reading(unit_id="chuong-1", at=NOW, values={"temperature": 35}),
                          Reading(unit_id="chuong-1", at=NOW + timedelta(hours=1), values={"temperature": 34})],
                         rules, "poultry")
    assert [a.rule_id for a in alerts] == ["po-nong-chuong"]
