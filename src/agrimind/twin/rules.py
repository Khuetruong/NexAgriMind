"""Bản sao số mức 1–2 (A08): phản ánh tình trạng và cảnh báo theo ngưỡng.

- Mức 1 (phản ánh): giá trị mới nhất và xu hướng của từng ao/lô/chuồng.
- Mức 2 (cảnh báo): so với ngưỡng trong tệp quy tắc.
- KHÔNG có mức tự điều khiển: mọi cảnh báo chỉ kèm đề xuất cho người quyết định (A06).
  Rủi ro mức 3 luôn ghi "cần chuyên gia xác nhận trước khi làm".
- Ngưỡng do chuyên gia đặt và ký duyệt; quy tắc chưa duyệt vẫn chạy nhưng câu
  cảnh báo ghi rõ "ngưỡng nháp".
"""

from __future__ import annotations

import operator
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

_OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


class Rule(BaseModel):
    id: str
    name_vi: str
    metric: str
    op: str = Field(pattern=r"^(<|<=|>|>=)$")
    threshold: float
    sectors: list[str] = Field(default_factory=list)  # rỗng = mọi ngành
    for_readings: int = Field(1, ge=1, le=100)  # số lần đo liên tiếp vượt ngưỡng
    risk_level: int = Field(ge=1, le=3)
    message: str
    suggestion: str
    source: str
    expert_approved: bool = False
    approved_by: str | None = None

    @model_validator(mode="after")
    def _approved(self) -> "Rule":
        if self.expert_approved and not self.approved_by:
            raise ValueError(f"{self.id}: đã duyệt thì phải ghi người duyệt")
        return self


class Reading(BaseModel):
    unit_id: str = Field(max_length=100)
    at: datetime
    values: dict[str, float]


class UnitState(BaseModel):
    unit_id: str
    last_at: datetime
    latest: dict[str, float]
    trend: dict[str, float]  # mới nhất − cũ nhất trong các lần đo gửi lên


class Alert(BaseModel):
    unit_id: str
    rule_id: str
    name_vi: str
    metric: str
    value: float
    threshold: float
    risk_level: int
    message: str
    suggestion: str
    action: str = "suggest_only"
    requires_expert_confirmation: bool
    rule_expert_approved: bool
    source: str


def load_rules(path: Path) -> list[Rule]:
    files = sorted(Path(path).glob("*.yaml")) if Path(path).is_dir() else [Path(path)]
    rules: list[Rule] = []
    for f in files:
        rules.extend(Rule(**r) for r in (yaml.safe_load(f.read_text(encoding="utf-8")) or {}).get("rules", []))
    ids = [r.id for r in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("trùng id quy tắc")
    return rules


def evaluate(readings: list[Reading], rules: list[Rule], sector: str | None = None) -> tuple[list[UnitState], list[Alert]]:
    by_unit: dict[str, list[Reading]] = {}
    for r in sorted(readings, key=lambda x: x.at):
        by_unit.setdefault(r.unit_id, []).append(r)
    states: list[UnitState] = []
    alerts: list[Alert] = []
    active = [r for r in rules if not r.sectors or (sector and sector in r.sectors)]
    for unit, rs in by_unit.items():
        latest: dict[str, float] = {}
        first: dict[str, float] = {}
        for r in rs:
            for k, v in r.values.items():
                first.setdefault(k, v)
                latest[k] = v
        states.append(UnitState(unit_id=unit, last_at=rs[-1].at, latest=latest,
                                trend={k: round(latest[k] - first[k], 4) for k in latest}))
        for rule in active:
            series = [r.values[rule.metric] for r in rs if rule.metric in r.values]
            tail = series[-rule.for_readings:]
            if len(tail) == rule.for_readings and all(_OPS[rule.op](v, rule.threshold) for v in tail):
                note = "" if rule.expert_approved else " (ngưỡng nháp, chờ chuyên gia duyệt)"
                alerts.append(Alert(
                    unit_id=unit, rule_id=rule.id, name_vi=rule.name_vi, metric=rule.metric, value=tail[-1],
                    threshold=rule.threshold, risk_level=rule.risk_level, message=rule.message + note,
                    suggestion=rule.suggestion, requires_expert_confirmation=rule.risk_level >= 3,
                    rule_expert_approved=rule.expert_approved, source=rule.source,
                ))
    alerts.sort(key=lambda a: (-a.risk_level, a.unit_id))
    return states, alerts
