"""Kho mô hình và cửa duyệt phát hành (A06).

Một mô hình chỉ được dùng cho người thật khi qua cửa duyệt:
1. Giấy phép cho dùng thương mại (A04): không AGPL, không phi thương mại.
2. Đánh giá trên tập tách THEO NHÓM (trang trại, mùa vụ hoặc vùng), không chia ngẫu nhiên
   từng dòng, để không "học thuộc" chính trại đó rồi tự chấm điểm cao.
3. Mọi chỉ số chính phải tốt hơn mốc so sánh (baseline) ít nhất một khoảng tối thiểu.
4. Chuyên gia ký duyệt: rủi ro mức 1 cần 1 người, mức 2 cần 1 người đúng lĩnh vực,
   mức 3 cần 2 người và mô hình không được tự hành động (chỉ đề xuất).
5. Mô hình học từ dữ liệu trang trại thì KHÔNG được phát hành lên Core hay sang nước khác (A07).

Trạng thái: draft → approved → published → retired. Lịch sử chỉ thêm, không sửa.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

COMMERCIAL_LICENSES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "cc0-1.0", "internal"}
BLOCKED_LICENSES = {"agpl-3.0", "gpl-3.0", "cc-by-nc-4.0", "cc-by-nc-sa-4.0", "cc-by-nd-4.0", "unknown"}


class Status(str, Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    PUBLISHED = "published"
    RETIRED = "retired"


class SplitBy(str, Enum):
    FARM = "farm"
    SEASON = "season"
    REGION = "region"
    TIME = "time"  # dự báo: huấn luyện quá khứ, kiểm tra tương lai
    RANDOM = "random"  # chỉ để ghi nhận; KHÔNG qua được cửa


class Metric(BaseModel):
    name: str
    value: float
    baseline: float
    higher_is_better: bool = True
    min_gain: float = 0.0  # khoảng hơn mốc tối thiểu (cùng đơn vị với chỉ số)
    primary: bool = True

    def beats_baseline(self) -> bool:
        gain = self.value - self.baseline if self.higher_is_better else self.baseline - self.value
        return gain > self.min_gain if self.min_gain > 0 else gain > 0


class Approval(BaseModel):
    expert: str
    domain: str  # mã trung tâm D01..D10
    at: datetime
    note: str = ""


class Event(BaseModel):
    at: datetime
    action: str
    by: str
    detail: str = ""


class ModelCard(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,80}$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    domain: str = Field(pattern=r"^D(0[1-9]|10)$")
    task: str
    license: str
    base_model: str | None = None
    trained_on_farm_data: bool = False
    country: str | None = None  # bắt buộc khi trained_on_farm_data
    datasets: list[str] = Field(default_factory=list)
    split_by: SplitBy
    metrics: list[Metric]
    risk_level: int = Field(ge=1, le=3)
    auto_action: bool = False  # True = tự điều khiển thiết bị/cho thuốc — không bao giờ cho phép ở mức 3
    artifact_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    artifact_uri: str | None = None
    # Cho mô hình chạy trên bộ AI tại trại: định dạng tệp và danh sách nhãn theo thứ tự lớp.
    runtime: str | None = Field(None, pattern=r"^(onnx|tflite|openvino)$")
    labels: list[str] = Field(default_factory=list, max_length=1000)
    status: Status = Status.DRAFT
    approvals: list[Approval] = Field(default_factory=list)
    history: list[Event] = Field(default_factory=list)

    @field_validator("license")
    @classmethod
    def _lic(cls, v: str) -> str:
        return v.strip().lower()

    @property
    def key(self) -> str:
        return f"{self.id}@{self.version}"


def gate_problems(card: ModelCard) -> list[str]:
    """Danh sách lý do chưa qua cửa duyệt; rỗng = đạt."""
    p: list[str] = []
    if card.license in BLOCKED_LICENSES or card.license not in COMMERCIAL_LICENSES:
        p.append(f"giấy phép '{card.license}' không được dùng cho sản phẩm")
    if card.split_by == SplitBy.RANDOM:
        p.append("đánh giá chia ngẫu nhiên; phải tách theo trang trại, mùa vụ, vùng hoặc thời gian")
    primary = [m for m in card.metrics if m.primary]
    if not primary:
        p.append("chưa có chỉ số chính so với mốc")
    for m in primary:
        if not m.beats_baseline():
            p.append(f"chỉ số {m.name} = {m.value} chưa tốt hơn mốc {m.baseline} (cần hơn {m.min_gain})")
    if card.trained_on_farm_data and not card.country:
        p.append("mô hình học từ dữ liệu trang trại phải ghi nước của dữ liệu")
    if card.risk_level == 3 and card.auto_action:
        p.append("rủi ro mức 3 không được tự hành động; chỉ được đề xuất")
    if not card.artifact_sha256:
        p.append("thiếu mã băm tệp mô hình")
    experts = {a.expert for a in card.approvals}
    in_domain = {a.expert for a in card.approvals if a.domain == card.domain}
    need = {1: 1, 2: 1, 3: 2}[card.risk_level]
    if len(experts) < need:
        p.append(f"rủi ro mức {card.risk_level} cần {need} chuyên gia ký, mới có {len(experts)}")
    if card.risk_level >= 2 and not in_domain:
        p.append(f"cần ít nhất một chuyên gia lĩnh vực {card.domain} ký")
    return p


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class RegistryError(ValueError):
    pass


class ModelRegistry:
    """Lưu mỗi phiên bản mô hình thành một tệp JSON trong thư mục."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, mid: str, version: str) -> Path:
        return self.root / f"{mid}@{version}.json"

    def get(self, mid: str, version: str) -> ModelCard:
        p = self._path(mid, version)
        if not p.exists():
            raise RegistryError(f"không có mô hình {mid}@{version}")
        return ModelCard.model_validate_json(p.read_text(encoding="utf-8"))

    def list(self) -> list[ModelCard]:
        return [ModelCard.model_validate_json(p.read_text(encoding="utf-8")) for p in sorted(self.root.glob("*.json"))]

    def _save(self, card: ModelCard) -> None:
        self._path(card.id, card.version).write_text(card.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def register(self, card: ModelCard, by: str) -> ModelCard:
        if self._path(card.id, card.version).exists():
            raise RegistryError(f"{card.key} đã có; phiên bản mới phải tăng số phiên bản")
        card = card.model_copy(update={"status": Status.DRAFT, "approvals": [], "history": []})
        card.history.append(Event(at=self._now(), action="register", by=by))
        self._save(card)
        return card

    def approve(self, mid: str, version: str, expert: str, domain: str, note: str = "") -> ModelCard:
        card = self.get(mid, version)
        if card.status != Status.DRAFT:
            raise RegistryError(f"{card.key} đang ở trạng thái {card.status.value}, không ký thêm được")
        if any(a.expert == expert for a in card.approvals):
            raise RegistryError(f"{expert} đã ký {card.key}")
        card.approvals.append(Approval(expert=expert, domain=domain, at=self._now(), note=note))
        card.history.append(Event(at=self._now(), action="approve", by=expert, detail=domain))
        if not gate_problems(card):
            card.status = Status.APPROVED
            card.history.append(Event(at=self._now(), action="gate_passed", by="system"))
        self._save(card)
        return card

    def mark_published(self, mid: str, version: str, by: str, target: str) -> ModelCard:
        card = self.get(mid, version)
        if card.status not in {Status.APPROVED, Status.PUBLISHED}:
            raise RegistryError(f"{card.key} chưa qua cửa duyệt")
        if card.trained_on_farm_data and target.upper() != (card.country or "").upper():
            raise RegistryError(f"{card.key} học từ dữ liệu trang trại {card.country}; không được gửi tới {target}")
        card.status = Status.PUBLISHED
        card.history.append(Event(at=self._now(), action="publish", by=by, detail=target))
        self._save(card)
        return card

    def retire(self, mid: str, version: str, by: str, reason: str) -> ModelCard:
        card = self.get(mid, version)
        card.status = Status.RETIRED
        card.history.append(Event(at=self._now(), action="retire", by=by, detail=reason))
        self._save(card)
        return card


def canonical(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
