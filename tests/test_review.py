import csv
from pathlib import Path

from agrimind.corpus.ingest import build_chunks, load_directory
from agrimind.eval.benchmark import load_benchmark
from agrimind.eval.review import export, report
from agrimind.kg.store import KnowledgeGraph
from agrimind.rag.answer import Assistant
from agrimind.rag.llm import StubLLM
from agrimind.rag.retriever import BM25Retriever

ROOT = Path(__file__).resolve().parents[1]


def _sheet(tmp_path):
    a = Assistant(BM25Retriever(build_chunks(load_directory(ROOT / "tests/fixtures/corpus"))), StubLLM(),
                  graph=KnowledgeGraph.from_dir(ROOT / "data/kg"))
    out = tmp_path / "cham.csv"
    assert export(a, load_benchmark(ROOT / "benchmarks/vi-core-v0.yaml"), out) == 6
    with out.open(encoding="utf-8-sig") as f:
        return out, list(csv.DictReader(f))


def _write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_export_has_answers_and_kg(tmp_path):
    _, rows = _sheet(tmp_path)
    ga = next(r for r in rows if r["id"] == "d03-002")
    assert ga["tu_choi"] == "khong" and "Tỷ lệ đẻ giảm" in ga["kha_nang_tu_do_thi"] and ga["dung"] == ""


def test_gate_report(tmp_path):
    out, rows = _sheet(tmp_path)
    rep = report(out, min_graded=5)
    assert rep.graded == 0 and not rep.passed and "mới có 0 câu" in rep.reasons()[0]
    for r in rows:
        r.update(chuyen_gia="TS. A", dung="dung", an_toan="co", nguon_phu_hop="co")
    _write(out, rows)
    assert report(out, min_graded=5).passed
    rows[0]["an_toan"] = "khong"
    rows[1]["dung"] = "dung roi"
    _write(out, rows)
    rep = report(out, min_graded=5)
    assert not rep.passed and rep.unsafe == [rows[0]["id"]] and rep.invalid == [rows[1]["id"]]
