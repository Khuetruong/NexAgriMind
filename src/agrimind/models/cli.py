"""agrimind-models: đăng ký, ký duyệt, ký phát hành và cài mô hình.

  agrimind-models register the.json --by <người>         # thẻ mô hình JSON
  agrimind-models check <id> <phiên bản>                  # lý do chưa qua cửa
  agrimind-models approve <id> <phiên bản> --expert <tên> --domain D02
  agrimind-models keygen --out core-signing               # chỉ chạy ở Core, giữ kín khoá .key
  agrimind-models publish <id> <phiên bản> --target VN --key core-signing.key --out goi.json
  agrimind-models install goi.json --pub core-signing.pub --country VN --artifact model.bin
Thư mục kho: AGRIMIND_MODEL_DIR (mặc định ./models-registry).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .registry import ModelCard, ModelRegistry, RegistryError, gate_problems
from .signing import BundleError, new_keypair, sign_bundle, verify_bundle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agrimind-models")
    ap.add_argument("--dir", type=Path, default=Path(os.getenv("AGRIMIND_MODEL_DIR", "models-registry")))
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register"); r.add_argument("card", type=Path); r.add_argument("--by", required=True)
    c = sub.add_parser("check"); c.add_argument("id"); c.add_argument("version")
    a = sub.add_parser("approve"); a.add_argument("id"); a.add_argument("version")
    a.add_argument("--expert", required=True); a.add_argument("--domain", required=True); a.add_argument("--note", default="")
    k = sub.add_parser("keygen"); k.add_argument("--out", required=True)
    p = sub.add_parser("publish"); p.add_argument("id"); p.add_argument("version")
    p.add_argument("--target", required=True); p.add_argument("--key", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--by", default="core")
    i = sub.add_parser("install"); i.add_argument("bundle", type=Path); i.add_argument("--pub", type=Path, required=True)
    i.add_argument("--country", required=True); i.add_argument("--artifact", type=Path)
    args = ap.parse_args(argv)
    reg = ModelRegistry(args.dir)
    try:
        if args.cmd == "register":
            card = reg.register(ModelCard.model_validate_json(args.card.read_text(encoding="utf-8")), args.by)
            print(f"Đã đăng ký {card.key} (bản nháp)")
        elif args.cmd == "check":
            probs = gate_problems(reg.get(args.id, args.version))
            print("ĐẠT cửa duyệt" if not probs else "CHƯA ĐẠT:\n" + "\n".join(f"  - {x}" for x in probs))
            return 0 if not probs else 1
        elif args.cmd == "approve":
            card = reg.approve(args.id, args.version, args.expert, args.domain, args.note)
            print(f"{card.key}: {card.status.value}")
            for x in gate_problems(card):
                print(f"  còn thiếu: {x}")
        elif args.cmd == "keygen":
            priv, pub = new_keypair()
            Path(args.out + ".key").write_bytes(priv)
            os.chmod(args.out + ".key", 0o600)
            Path(args.out + ".pub").write_bytes(pub)
            print(f"Khoá bí mật: {args.out}.key (giữ kín, chỉ ở Core). Khoá công khai: {args.out}.pub (gửi các nước).")
        elif args.cmd == "publish":
            card = reg.get(args.id, args.version)
            bundle = sign_bundle(card, args.target, args.key.read_bytes())
            reg.mark_published(args.id, args.version, args.by, args.target)
            args.out.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Đã ký gói {card.key} cho {args.target.upper()}: {args.out}")
        elif args.cmd == "install":
            card = verify_bundle(json.loads(args.bundle.read_text(encoding="utf-8")), args.pub.read_bytes(),
                                 args.country, args.artifact)
            dest = reg.root / f"{card.key}.json"
            dest.write_text(card.model_dump_json(indent=2), encoding="utf-8")
            print(f"Chữ ký đúng. Đã cài {card.key} vào {reg.root}")
    except (RegistryError, BundleError, ValueError, OSError) as e:
        print(f"LỖI: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
