"""agrimind-eval --corpus <thư mục> --bench <file.yaml> [--llm-url URL --model TÊN]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..corpus.ingest import build_chunks, load_directory
from ..rag.answer import Assistant
from ..rag.llm import OpenAICompatibleLLM, StubLLM
from ..rag.retriever import BM25Retriever
from .benchmark import load_benchmark, run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agrimind-eval")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--llm-url", default="")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    args = ap.parse_args(argv)
    llm = OpenAICompatibleLLM(args.llm_url, args.model) if args.llm_url else StubLLM()
    assistant = Assistant(BM25Retriever(build_chunks(load_directory(args.corpus))), llm)
    rep = run(assistant, load_benchmark(args.bench))
    print(json.dumps(rep.summary(), ensure_ascii=False, indent=2))
    for r in rep.results:
        if not r.passed:
            print(f"KHÔNG ĐẠT {r.id}: {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
