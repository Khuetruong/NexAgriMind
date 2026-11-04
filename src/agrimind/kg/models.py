"""Đồ thị tri thức (A03, A05).

Quan hệ phải có hướng tác động, điều kiện, bằng chứng và mức chắc chắn.
Không được coi tương quan là nhân quả: loại quan hệ CORRELATES_WITH
không bao giờ được dùng như CAUSES khi lập luận.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class EntityType(str, Enum):
    SPECIES = "species"
    VARIETY = "variety"
    GENE = "gene"
    MICROBE = "microbe"
    ENZYME = "enzyme"
    NUTRIENT = "nutrient"
    COMPOUND = "compound"
    DISEASE = "disease"
    PATHOGEN = "pathogen"
    ENV_FACTOR = "env_factor"
    PRACTICE = "practice"
    BIO_PRODUCT = "bio_product"
    OUTCOME = "outcome"


class Entity(BaseModel):
    id: str
    type: EntityType
    name_vi: str
    name_en: str | None = None
    scientific_name: str | None = None
    domains: list[str] = Field(default_factory=list)


class RelationType(str, Enum):
    CAUSES = "causes"
    INCREASES_RISK_OF = "increases_risk_of"
    DECREASES_RISK_OF = "decreases_risk_of"
    INCREASES = "increases"
    DECREASES = "decreases"
    CORRELATES_WITH = "correlates_with"  # chỉ tương quan, không phải nhân quả
    REQUIRES = "requires"
    PART_OF = "part_of"

    @property
    def causal(self) -> bool:
        return self not in {RelationType.CORRELATES_WITH, RelationType.PART_OF}


class EvidenceGrade(str, Enum):
    """Thang bằng chứng A–E của A05."""

    A = "A"  # tổng quan hệ thống / phân tích gộp
    B = "B"  # thí nghiệm có đối chứng
    C = "C"  # nghiên cứu quan sát
    D = "D"  # ý kiến chuyên gia, báo cáo ca
    E = "E"  # chưa kiểm chứng, giả thuyết


class Certainty(str, Enum):
    """Bốn mức tin cậy của A05."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class Evidence(BaseModel):
    source: str = Field(description="Trích dẫn hoặc id tài liệu trong kho")
    grade: EvidenceGrade
    note: str | None = None


class Relation(BaseModel):
    id: str
    subject: str
    type: RelationType
    object: str
    conditions: list[str] = Field(default_factory=list, description="Điều kiện để quan hệ đúng")
    evidence: list[Evidence]
    certainty: Certainty
    expert_approved: bool = False
    approved_by: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "Relation":
        if not self.evidence:
            raise ValueError(f"{self.id}: quan hệ phải có ít nhất một bằng chứng")
        if self.subject == self.object:
            raise ValueError(f"{self.id}: chủ thể và đối tượng trùng nhau")
        if self.expert_approved and not self.approved_by:
            raise ValueError(f"{self.id}: đã duyệt thì phải ghi người duyệt")
        best = min(e.grade.value for e in self.evidence)
        if self.certainty == Certainty.HIGH and best not in {"A", "B"}:
            raise ValueError(f"{self.id}: mức 'high' cần bằng chứng A hoặc B")
        if self.type == RelationType.CORRELATES_WITH and self.certainty == Certainty.HIGH:
            # tương quan mạnh vẫn chỉ là tương quan; giữ mức tối đa 'medium' để tránh hiểu nhầm
            raise ValueError(f"{self.id}: quan hệ tương quan không được gắn mức 'high'")
        return self
