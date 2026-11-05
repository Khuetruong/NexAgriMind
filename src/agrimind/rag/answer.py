"""Trợ lý hỏi đáp có dẫn nguồn, trả lời theo mẫu 6 phần của A05.

Sáu phần: kết luận · bằng chứng (có nguồn) · độ tin cậy · giả định ·
cách kiểm chứng · việc cần chuyên gia xác nhận.

Quy tắc an toàn được kiểm tra bằng mã, không chỉ dặn trong lời nhắc:
- không tìm được tài liệu đủ liên quan → từ chối, xin thêm dữ liệu;
- nguồn mô hình trích phải nằm trong các đoạn đã đưa cho nó, nguồn bịa bị loại;
- không còn bằng chứng hợp lệ → hạ độ tin cậy về 'insufficient'.
"""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, Field

from ..kg.models import Certainty
from ..kg.store import KnowledgeGraph
from .llm import LLM
from .retriever import Hit

SYSTEM_PROMPT = """Bạn là trợ lý khoa học nông nghiệp AgriMind.
Chỉ dùng các nguồn được cung cấp, đánh dấu [S1], [S2]...
Không kết luận vội: liệt kê các khả năng, nói rõ giả định.
Tương quan không phải nhân quả.
Không đưa liều thuốc, hóa chất cụ thể; việc đó cần kỹ sư hoặc bác sĩ thú y xác nhận.
Nếu nguồn không đủ, đặt confidence = "insufficient" và nói cần thêm dữ liệu gì.
Trả về JSON với các khóa: conclusion (chuỗi), evidence (danh sách {claim, source}),
confidence (high|medium|low|insufficient), assumptions (danh sách), validation (danh sách),
expert_check (danh sách). Trả lời bằng tiếng Việt."""


class Retriever(Protocol):
    def search(self, query: str, k: int = 5, domains: list[str] | None = None) -> list[Hit]: ...


class PossibleCause(BaseModel):
    """Một khả năng lấy từ đồ thị tri thức (chỉ quan hệ nhân quả, không lấy tương quan)."""

    outcome: str
    factor: str
    relation: str
    certainty: Certainty
    conditions: list[str] = Field(default_factory=list)
    expert_approved: bool = False


class EvidenceItem(BaseModel):
    claim: str
    source_id: str
    source: str
    title: str


class Answer(BaseModel):
    question: str
    conclusion: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: Certainty
    assumptions: list[str] = Field(default_factory=list)
    validation: list[str] = Field(default_factory=list)
    expert_check: list[str] = Field(default_factory=list)
    refused: bool = False
    dropped_citations: int = 0
    # True khi dữ liệu trang trại (đã được trại đồng ý chia sẻ) có trong lời nhắc.
    farm_data_used: bool = False
    # local = kho trong nước; core = Core trả lời qua trạm kiểm soát dữ liệu ra
    answered_by: str = "local"
    # Các khả năng cần loại trừ, từ đồ thị tri thức (A05: liệt kê khả năng trước khi kết luận)
    possible_causes: list[PossibleCause] = Field(default_factory=list)


class Assistant:
    def __init__(self, retriever: Retriever, llm: LLM, k: int = 5, min_score: float = 1.0,
                 graph: KnowledgeGraph | None = None) -> None:
        self.graph = graph
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.min_score = min_score

    def _refuse(self, question: str, reason: str) -> Answer:
        return Answer(
            question=question,
            conclusion="Chưa đủ căn cứ để trả lời. " + reason,
            confidence=Certainty.INSUFFICIENT,
            validation=["Bổ sung dữ liệu hoặc tài liệu liên quan rồi hỏi lại."],
            expert_check=["Chuyển câu hỏi cho chuyên gia của trung tâm phụ trách."],
            refused=True,
        )

    def possible_causes(self, question: str) -> list[PossibleCause]:
        if not self.graph:
            return []
        out: list[PossibleCause] = []
        for outcome in self.graph.outcomes_in(question):
            for r in self.graph.candidate_causes(outcome.id):
                factor = self.graph.entities[r.subject]
                out.append(PossibleCause(
                    outcome=outcome.name_vi, factor=factor.name_vi, relation=r.type.value,
                    certainty=r.certainty, conditions=r.conditions, expert_approved=r.expert_approved,
                ))
        return out[:12]

    def ask(self, question: str, domains: list[str] | None = None, farm_context: dict | None = None) -> Answer:
        hits = self.retriever.search(question, k=self.k, domains=domains)
        hits = [h for h in hits if (h.relevant if h.relevant is not None else h.score >= self.min_score)]
        if not hits:
            return self._refuse(question, "Kho tài liệu chưa có nguồn đủ liên quan.")

        sources = {f"S{i + 1}": h for i, h in enumerate(hits)}
        user = "Câu hỏi: " + question + "\n\nNguồn:\n" + "\n".join(
            f"[{sid}] {h.chunk.title}: {h.chunk.text}" for sid, h in sources.items()
        )
        causes = self.possible_causes(question)
        if causes:
            # Đồ thị chỉ gợi ý khả năng cần xét; bằng chứng vẫn phải là nguồn [S].
            user += "\n\nCác khả năng theo đồ thị tri thức (gợi ý để xét, KHÔNG phải nguồn để trích):\n" + "\n".join(
                f"- {c.factor} → {c.outcome} ({c.relation}, độ chắc chắn {c.certainty.value})" for c in causes
            )
        if farm_context:
            # Dữ liệu trại chỉ để hiểu bối cảnh; không phải nguồn khoa học nên không có mã [S].
            user += (
                "\n\nDữ liệu trang trại (bối cảnh, KHÔNG phải nguồn khoa học, không trích làm [S]):\n"
                + json.dumps(farm_context, ensure_ascii=False, default=str)[:4000]
            )
        raw = self.llm.complete_json(SYSTEM_PROMPT, user)
        ans = self._validate(question, raw, sources)
        ans.farm_data_used = bool(farm_context)
        ans.possible_causes = causes
        return ans

    @staticmethod
    def _validate(question: str, raw: dict, sources: dict[str, Hit]) -> Answer:
        evidence: list[EvidenceItem] = []
        dropped = 0
        for item in raw.get("evidence") or []:
            sid = str((item or {}).get("source", "")).strip("[] ")
            hit = sources.get(sid)
            if hit is None:
                dropped += 1
                continue
            evidence.append(
                EvidenceItem(
                    claim=str(item.get("claim", "")),
                    source_id=hit.chunk.id,
                    source=hit.chunk.source,
                    title=hit.chunk.title,
                )
            )
        try:
            confidence = Certainty(raw.get("confidence", "insufficient"))
        except ValueError:
            confidence = Certainty.INSUFFICIENT
        if not evidence:
            confidence = Certainty.INSUFFICIENT

        def as_list(key: str) -> list[str]:
            v = raw.get(key) or []
            return [str(x) for x in (v if isinstance(v, list) else [v])]

        return Answer(
            question=question,
            conclusion=str(raw.get("conclusion", "")).strip() or "Chưa đủ căn cứ để kết luận.",
            evidence=evidence,
            confidence=confidence,
            assumptions=as_list("assumptions"),
            validation=as_list("validation"),
            expert_check=as_list("expert_check"),
            refused=confidence == Certainty.INSUFFICIENT,
            dropped_citations=dropped,
        )
