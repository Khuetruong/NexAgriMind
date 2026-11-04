"""agrimind-ingest <thư mục>: kiểm tra tài liệu và in thống kê giấy phép."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .ingest import IngestError, build_chunks, load_directory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agrimind-ingest")
    ap.add_argument("root", type=Path)
    args = ap.parse_args(argv)
    try:
        docs = load_directory(args.root)
    except IngestError as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    lic = Counter(d.license.value for d in docs)
    chunks = build_chunks(docs)
    print(f"Tài liệu: {len(docs)}")
    for k, v in sorted(lic.items()):
        print(f"  {k}: {v}")
    print(f"Đoạn dùng được cho sản phẩm (giấy phép thương mại): {len(chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
