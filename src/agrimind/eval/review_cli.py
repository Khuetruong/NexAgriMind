"""agrimind-review export|report — chuyên gia chấm câu trả lời, xem cửa giai đoạn 1."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..corpus.ingest import build_chunks, load_directory
from ..kg.store import KnowledgeGraph
from ..rag.answer import Assistant
from ..rag.llm import OpenAICompatibleLLM, StubLLM
from ..rag.retriever import BM25Retriever
from .benchmark import load_benchmark
from .review import export, report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agrimind-review")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="chạy trợ lý, ghi bảng cho chuyên gia chấm")
    e.add_argument("--corpus", type=Path, required=True)
    e.add_argument("--bench", type=Path, required=True)
    e.add_argument("--kg", type=Path, default=Path("data/kg"))
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--llm-url", default=os.getenv("AGRIMIND_LLM_URL", ""))
    e.add_argument("--model", default=os.getenv("AGRIMIND_LLM_MODEL", "Qwen/Qwen3-8B"))
    r = sub.add_parser("report", help="đọc bảng đã chấm, báo cáo cửa giai đoạn")
    r.add_argument("sheet", type=Path)
    r.add_argument("--min-graded", type=int, default=30)
    r.add_argument("--min-correct", type=float, default=0.8)
    args = ap.parse_args(argv)

    if args.cmd == "export":
        llm = OpenAICompatibleLLM(args.llm_url, args.model) if args.llm_url else StubLLM()
        graph = KnowledgeGraph.from_dir(args.kg) if args.kg.is_dir() else None
        a = Assistant(BM25Retriever(build_chunks(load_directory(args.corpus))), llm, graph=graph)
        n = export(a, load_benchmark(args.bench), args.out)
        print(f"Đã ghi {n} câu vào {args.out}. Gửi chuyên gia điền các cột chuyen_gia, dung, an_toan, "
              f"nguon_phu_hop, ghi_chu.")
        return 0

    rep = report(args.sheet, args.min_graded, args.min_correct)
    print(f"Đã chấm {rep.graded}/{rep.total} câu; tỷ lệ đúng {rep.correct_rate:.0%}")
    for d, (n, ok) in sorted(rep.by_domain.items()):
        print(f"  {d}: {ok}/{n} đúng hoàn toàn")
    if rep.bad_sources:
        print(f"  Nguồn không phù hợp: {', '.join(rep.bad_sources)}")
    if rep.passed:
        print("CỬA GIAI ĐOẠN 1 (phần trợ lý): ĐẠT")
        return 0
    print("CỬA GIAI ĐOẠN 1 (phần trợ lý): CHƯA ĐẠT")
    for x in rep.reasons():
        print(f"  - {x}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
