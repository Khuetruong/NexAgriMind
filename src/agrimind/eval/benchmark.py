"""Bộ chấm điểm tự động (A06 – bộ kiểm tra chuẩn).

Mỗi câu hỏi chuẩn có:
  id, question, domains, expect: answerable | refuse,
  expected_sources: id tài liệu phải được trích (khi answerable),
  status: draft | expert_approved, reviewer.

Chấm tự động 4 tiêu chí; độ đúng khoa học của kết luận do chuyên gia chấm tay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ..rag.answer import Answer, Assistant


class BenchItem(BaseModel):
    id: str
    question: str
    domains: list[str] | None = None
    expect: str = Field(pattern="^(answerable|refuse)$")
    expected_sources: list[str] = Field(default_factory=list)
    status: str = "draft"
    reviewer: str | None = None


def load_benchmark(path: Path) -> list[BenchItem]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [BenchItem(**x) for x in data.get("items", [])]


@dataclass
class ItemResult:
    id: str
    refusal_correct: bool
    cites_expected: bool | None
    template_complete: bool
    no_fabricated_citations: bool

    @property
    def passed(self) -> bool:
        return (
            self.refusal_correct
            and self.cites_expected is not False
            and self.template_complete
            and self.no_fabricated_citations
        )


@dataclass
class Report:
    results: list[ItemResult] = field(default_factory=list)

    def rate(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.results if getattr(r, attr) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict:
        return {
            "items": len(self.results),
            "passed": sum(r.passed for r in self.results),
            "refusal_correct": round(self.rate("refusal_correct"), 3),
            "cites_expected": round(self.rate("cites_expected"), 3),
            "template_complete": round(self.rate("template_complete"), 3),
            "no_fabricated_citations": round(self.rate("no_fabricated_citations"), 3),
        }


def score(item: BenchItem, ans: Answer) -> ItemResult:
    refusal_correct = ans.refused == (item.expect == "refuse")
    cites = None
    if item.expect == "answerable" and item.expected_sources:
        got = {e.source_id.split("#")[0] for e in ans.evidence}
        cites = bool(got & set(item.expected_sources))
    complete = bool(ans.conclusion) and (ans.refused or (bool(ans.evidence) and bool(ans.validation)))
    return ItemResult(item.id, refusal_correct, cites, complete, ans.dropped_citations == 0)


def run(assistant: Assistant, items: list[BenchItem]) -> Report:
    rep = Report()
    for it in items:
        rep.results.append(score(it, assistant.ask(it.question, domains=it.domains)))
    return rep
