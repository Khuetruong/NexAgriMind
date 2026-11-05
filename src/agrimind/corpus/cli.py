"""agrimind-ingest <thư mục>: kiểm tra tài liệu và in thống kê giấy phép.
agrimind-fetch <danh sách nguồn.yaml> --out <thư mục>: tải tài liệu mở về kho."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .ingest import IngestError, build_chunks, load_directory
from .sources import fetch_manifest


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


def fetch_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agrimind-fetch")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rep = fetch_manifest(args.manifest, args.out)
    print(f"Đã ghi: {len(rep.written)}, không đổi: {len(rep.unchanged)}, lỗi: {len(rep.errors)}")
    for sid in rep.unknown_license:
        print(f"  {sid}: giấy phép không cho dùng thương mại hoặc chưa rõ → không vào kho sản phẩm")
    for sid, err in rep.errors.items():
        print(f"  LỖI {sid}: {err}", file=sys.stderr)
    return 1 if rep.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
