"""agrimind-keys new <mã bên> --scopes ... : tạo khoá API mới.

In khoá MỘT LẦN (giao cho bên kia qua kênh an toàn) và đoạn cấu hình chỉ chứa
mã băm để dán vào tệp AGRIMIND_CLIENTS_FILE.
"""

from __future__ import annotations

import argparse
import sys

import yaml

from .clients import Scope, new_key


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agrimind-keys")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("client_id")
    n.add_argument("--name", default="")
    n.add_argument("--scopes", default=Scope.ASK_GENERAL.value,
                   help="cách nhau bằng dấu phẩy: " + ", ".join(s.value for s in Scope))
    n.add_argument("--countries", default="", help="ví dụ VN,TH; bỏ trống = không giới hạn")
    n.add_argument("--rate", type=int, default=60, help="số yêu cầu mỗi phút")
    args = ap.parse_args(argv)
    try:
        scopes = sorted({Scope(s.strip()).value for s in args.scopes.split(",") if s.strip()})
        key, sha = new_key(args.client_id)
    except ValueError as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 2
    entry = {"id": args.client_id, "name": args.name or args.client_id, "key_sha256": sha,
             "scopes": scopes, "rate_per_minute": args.rate}
    if args.countries:
        entry["countries"] = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    print("Khoá (chỉ hiện một lần, gửi cho bên kia qua kênh an toàn):", file=sys.stderr)
    print(key, file=sys.stderr)
    print("\nDán vào tệp cấu hình các bên (AGRIMIND_CLIENTS_FILE):", file=sys.stderr)
    print(yaml.safe_dump({"clients": [entry]}, allow_unicode=True, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
