"""Trợ lý hỏi đáp có dẫn nguồn, trả lời theo mẫu 6 phần của A05.

Sáu phần: kết luận · bằng chứng (có nguồn) · độ tin cậy · giả định ·
cách kiểm chứng · việc cần chuyên gia xác nhận.

Quy tắc an toàn được kiểm tra bằng mã, không chỉ dặn trong lời nhắc:
- không tìm được tài liệu đủ liên quan → từ chối, xin thêm dữ liệu;
- nguồn mô hình trích phải nằm trong các đoạn đã đưa cho nó, nguồn bịa bị loại;
- không còn bằng chứng hợp lệ → hạ độ tin cậy về 'insufficient'.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..kg.models import Certainty
from .llm import LLM
from .retriever import BM25Retriever, Hit

SYSTEM_PROMPT = """Bạn là trợ lý khoa học nông nghiệp AgriMind.
Chỉ dùng các nguồn được cung cấp, đánh dấu [S1], [S2]...
Không kết luận vội: liệt kê các khả năng, nói rõ giả định.
Tương quan không phải nhân quả.
Không đưa liều thuốc, hóa chất cụ thể; việc đó cần kỹ sư hoặc bác sĩ thú y xác nhận.
Nếu nguồn không đủ, đặt confidence = "insufficient" và nói cần thêm dữ liệu gì.
Trả về JSON với các khóa: conclusion (chuỗi), evidence (danh sách {claim, source}),
confidence (high|medium|low|insufficient), assumptions (danh sách), validation (danh sách),
expert_check (danh sách). Trả lời bằng tiếng Việt."""


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


class Assistant:
    def __init__(self, retriever: BM25Retriever, llm: LLM, k: int = 5, min_score: float = 1.0) -> None:
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

    def ask(self, question: str, domains: list[str] | None = None) -> Answer:
        hits = self.retriever.search(question, k=self.k, domains=domains)
        hits = [h for h in hits if h.score >= self.min_score]
        if not hits:
            return self._refuse(question, "Kho tài liệu chưa có nguồn đủ liên quan.")

        sources = {f"S{i + 1}": h for i, h in enumerate(hits)}
        user = "Câu hỏi: " + question + "\n\nNguồn:\n" + "\n".join(
            f"[{sid}] {h.chunk.title}: {h.chunk.text}" for sid, h in sources.items()
        )
        raw = self.llm.complete_json(SYSTEM_PROMPT, user)
        return self._validate(question, raw, sources)

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
