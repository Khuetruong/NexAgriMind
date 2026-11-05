"""agrimind-index <thư mục>: nhúng tài liệu (giấy phép thương mại) vào Qdrant."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

from ..corpus.ingest import IngestError, build_chunks, load_directory
from .vector import OpenAICompatibleEmbedder, QdrantStore, VectorRetriever


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agrimind-index")
    ap.add_argument("root", type=Path)
    ap.add_argument("--qdrant-url", default=os.getenv("AGRIMIND_QDRANT_URL", ""))
    ap.add_argument("--collection", default=os.getenv("AGRIMIND_QDRANT_COLLECTION", "agrimind_corpus"))
    ap.add_argument("--embed-url", default=os.getenv("AGRIMIND_EMBED_URL", ""))
    ap.add_argument("--embed-model", default=os.getenv("AGRIMIND_EMBED_MODEL", "BAAI/bge-m3"))
    args = ap.parse_args(argv)
    if not args.qdrant_url or not args.embed_url:
        print("Cần --qdrant-url và --embed-url (hoặc AGRIMIND_QDRANT_URL, AGRIMIND_EMBED_URL).", file=sys.stderr)
        return 2
    try:
        chunks = build_chunks(load_directory(args.root))
    except IngestError as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    vr = VectorRetriever(
        QdrantStore(args.qdrant_url, args.collection, os.getenv("AGRIMIND_QDRANT_API_KEY")),
        OpenAICompatibleEmbedder(args.embed_url, args.embed_model, os.getenv("AGRIMIND_EMBED_API_KEY")),
    )
    try:
        n = vr.index(chunks)
    except httpx.HTTPError as e:
        print(f"LỖI kết nối: {e}", file=sys.stderr)
        return 1
    print(f"Đã nhúng {n} đoạn vào {args.qdrant_url}/{args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
