"""agrimind-kg <file.yaml> [--neo4j-url ...]: kiểm tra đồ thị, in Cypher hoặc nạp vào Neo4j."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .neo4j import Neo4jError, load
from .store import KnowledgeGraph


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agrimind-kg")
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--neo4j-url", default=os.getenv("AGRIMIND_NEO4J_URL", ""))
    ap.add_argument("--neo4j-user", default=os.getenv("AGRIMIND_NEO4J_USER", "neo4j"))
    ap.add_argument("--database", default=os.getenv("AGRIMIND_NEO4J_DB", "neo4j"))
    ap.add_argument("--cypher", action="store_true", help="chỉ in câu lệnh Cypher")
    args = ap.parse_args(argv)
    for f in args.files:
        try:
            g = KnowledgeGraph.from_yaml(f)
        except (ValueError, OSError) as e:
            print(f"LỖI {f}: {e}", file=sys.stderr)
            return 1
        approved = sum(r.expert_approved for r in g.relations.values())
        print(f"{f}: {len(g.entities)} thực thể, {len(g.relations)} quan hệ ({approved} đã chuyên gia duyệt)")
        if args.cypher:
            print(g.to_cypher())
        elif args.neo4j_url:
            pw = os.getenv("AGRIMIND_NEO4J_PASSWORD", "")
            try:
                n = load(g, args.neo4j_url, args.neo4j_user, pw, args.database)
            except (Neo4jError, OSError) as e:
                print(f"LỖI nạp Neo4j: {e}", file=sys.stderr)
                return 1
            print(f"  đã nạp {n} câu lệnh vào {args.neo4j_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
