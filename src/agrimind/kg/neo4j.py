"""Nạp đồ thị tri thức vào Neo4j qua HTTP API (không cần driver riêng).

Dùng câu lệnh có tham số (không ghép chuỗi) để tên tiếng Việt, dấu nháy... không
làm hỏng câu lệnh. Nạp nhiều lần cho cùng kết quả (MERGE).
"""

from __future__ import annotations

import json

import httpx

from .store import KnowledgeGraph


class Neo4jError(RuntimeError):
    pass


def statements(g: KnowledgeGraph) -> list[dict]:
    out: list[dict] = [
        {"statement": "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE"}
    ]
    for e in g.entities.values():
        out.append({
            "statement": "MERGE (n:Entity {id: $id}) SET n.type = $type, n.name_vi = $name_vi, n.json = $json",
            "parameters": {
                "id": e.id, "type": e.type.value, "name_vi": e.name_vi,
                "json": json.dumps(e.model_dump(mode="json"), ensure_ascii=False),
            },
        })
    for r in g.relations.values():
        # Loại quan hệ không truyền tham số được trong Cypher; lấy từ enum nên an toàn.
        rel = r.type.value.upper()
        out.append({
            "statement": (
                "MATCH (a:Entity {id: $s}), (b:Entity {id: $o}) "
                f"MERGE (a)-[x:{rel} {{id: $id}}]->(b) "
                "SET x.causal = $causal, x.certainty = $certainty, x.approved = $approved, x.json = $json"
            ),
            "parameters": {
                "s": r.subject, "o": r.object, "id": r.id, "causal": r.type.causal,
                "certainty": r.certainty.value, "approved": r.expert_approved,
                "json": json.dumps(r.model_dump(mode="json"), ensure_ascii=False),
            },
        })
    return out


def load(g: KnowledgeGraph, url: str, user: str, password: str, database: str = "neo4j",
         client: httpx.Client | None = None) -> int:
    http = client or httpx.Client(timeout=60)
    stmts = statements(g)
    resp = http.post(
        f"{url.rstrip('/')}/db/{database}/tx/commit",
        auth=(user, password),
        json={"statements": stmts},
    )
    resp.raise_for_status()
    errors = resp.json().get("errors") or []
    if errors:
        raise Neo4jError("; ".join(f"{e.get('code')}: {e.get('message')}" for e in errors))
    return len(stmts)
