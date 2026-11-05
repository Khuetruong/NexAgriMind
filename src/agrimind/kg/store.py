"""Kho đồ thị trong bộ nhớ + xuất Cypher cho Neo4j."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import yaml

from .models import Entity, Relation


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())


class GraphError(ValueError):
    pass


class KnowledgeGraph:
    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: dict[str, Relation] = {}

    def add_entity(self, e: Entity) -> None:
        if e.id in self.entities:
            raise GraphError(f"trùng thực thể {e.id}")
        self.entities[e.id] = e

    def add_relation(self, r: Relation) -> None:
        for end in (r.subject, r.object):
            if end not in self.entities:
                raise GraphError(f"{r.id}: thực thể '{end}' chưa có")
        if r.id in self.relations:
            raise GraphError(f"trùng quan hệ {r.id}")
        self.relations[r.id] = r

    def outgoing(self, entity_id: str, causal_only: bool = False) -> list[Relation]:
        return [
            r
            for r in self.relations.values()
            if r.subject == entity_id and (not causal_only or r.type.causal)
        ]

    def incoming(self, entity_id: str, causal_only: bool = False) -> list[Relation]:
        return [
            r
            for r in self.relations.values()
            if r.object == entity_id and (not causal_only or r.type.causal)
        ]

    def candidate_causes(self, outcome_id: str) -> list[Relation]:
        """Các yếu tố có quan hệ nhân quả tới một kết quả (bỏ qua tương quan).

        Dùng cho bước 'liệt kê các khả năng' của A05, ví dụ tôm giảm ăn.
        """
        return sorted(
            self.incoming(outcome_id, causal_only=True),
            key=lambda r: (min(e.grade.value for e in r.evidence), r.id),
        )

    def outcomes_in(self, text: str) -> list[Entity]:
        """Các kết quả (outcome) được nhắc tới trong câu hỏi, so theo tên tiếng Việt."""
        words = set(_words(text))

        def hit(name: str) -> bool:
            w = _words(name)
            return bool(w) and set(w) <= words

        return [
            e for e in self.entities.values()
            if e.type.value == "outcome" and any(hit(n) for n in [e.name_vi, *e.aliases_vi])
        ]

    def merge(self, other: "KnowledgeGraph") -> None:
        for e in other.entities.values():
            if e.id in self.entities and self.entities[e.id] != e:
                raise GraphError(f"thực thể {e.id} khác nhau giữa hai tệp")
            self.entities.setdefault(e.id, e)
        for r in other.relations.values():
            self.add_relation(r)

    @classmethod
    def from_dir(cls, root: Path) -> "KnowledgeGraph":
        g = cls()
        for f in sorted(Path(root).glob("*.yaml")):
            g.merge(cls.from_yaml(f))
        return g

    @classmethod
    def from_yaml(cls, path: Path) -> "KnowledgeGraph":
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        g = cls()
        for e in data.get("entities", []):
            g.add_entity(Entity(**e))
        for r in data.get("relations", []):
            g.add_relation(Relation(**r))
        return g

    def to_cypher(self) -> str:
        """Câu lệnh MERGE idempotent để nạp vào Neo4j."""
        lines: list[str] = []
        for e in self.entities.values():
            props = json.dumps(e.model_dump(mode="json"), ensure_ascii=False)
            lines.append(
                "MERGE (n:Entity {id: %s}) SET n.type = %s, n.name_vi = %s, n.json = %s;"
                % (
                    json.dumps(e.id),
                    json.dumps(e.type.value),
                    json.dumps(e.name_vi, ensure_ascii=False),
                    json.dumps(props, ensure_ascii=False),
                )
            )
        for r in self.relations.values():
            props = json.dumps(r.model_dump(mode="json"), ensure_ascii=False)
            lines.append(
                "MATCH (a:Entity {id: %s}), (b:Entity {id: %s}) "
                "MERGE (a)-[x:%s {id: %s}]->(b) SET x.json = %s;"
                % (
                    json.dumps(r.subject),
                    json.dumps(r.object),
                    r.type.value.upper(),
                    json.dumps(r.id),
                    json.dumps(props, ensure_ascii=False),
                )
            )
        return "\n".join(lines)
