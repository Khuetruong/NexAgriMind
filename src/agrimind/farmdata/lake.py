"""Kho dữ liệu trang trại TRONG NƯỚC (A03, A07).

Chỉ điểm `country` có kho này. Quy tắc:
- Chỉ nhận dữ liệu của trại đã bật chia sẻ (hỏi lại Nexagri cho từng trại, mỗi lô).
- Bỏ nhận dạng trước khi lưu: mã trại, mã lô thay bằng mã giả (HMAC với khoá bí mật
  của điểm trong nước); toạ độ làm tròn còn 0,1 độ (khoảng 11 km); ghi chú tự do qua
  bộ lọc thông tin cá nhân; trường lạ bị bỏ.
- Kiểm tra chất lượng (mức raw → checked của A03): số hợp lệ, trong khoảng hợp lý,
  thời gian không ở tương lai. Không đạt vẫn lưu nhưng gắn cờ, không dùng huấn luyện.
- Trại yêu cầu xoá → xoá toàn bộ dữ liệu của mã giả đó.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import shutil
import threading
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from ..gateway.routing import redact


class RecordKind(str, Enum):
    TELEMETRY = "telemetry"
    WATER_LOG = "water_log"
    FEEDING_LOG = "feeding_log"
    INPUT_USE = "input_use"
    HARVEST = "harvest"
    OBSERVATION = "observation"
    VISION_EVENT = "vision_event"  # camera/hộp AI phát hiện (bệnh, sâu, đếm con, xâm nhập...)


class VisionInfo(BaseModel):
    """Kết quả AI trên ảnh. Nhãn người duyệt là dữ liệu quý nhất để huấn luyện lại."""

    camera: str | None = Field(None, max_length=100)  # mã camera (sẽ đổi thành mã giả)
    event_type: str = Field(max_length=60)
    predicted_label: str = Field(max_length=120)
    confidence: float = Field(ge=0, le=1)
    model_name: str | None = Field(None, max_length=120)
    model_version: str | None = Field(None, max_length=120)
    reviewed_label: str | None = Field(None, max_length=120)  # người duyệt sửa/ xác nhận
    review_verdict: str | None = Field(None, pattern="^(confirmed|rejected|corrected)$")


class FarmRecord(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=100)
    kind: RecordKind
    at: datetime
    plot_id: str | None = Field(None, max_length=100)
    sector: str | None = Field(None, max_length=40)  # crop | livestock | poultry | aquaculture | ...
    species: str | None = Field(None, max_length=100)
    province: str | None = Field(None, max_length=60)
    lat: float | None = None
    lon: float | None = None
    values: dict[str, float] = Field(default_factory=dict, max_length=100)
    units: dict[str, str] = Field(default_factory=dict, max_length=100)
    note: str | None = Field(None, max_length=2000)
    vision: VisionInfo | None = None
    source_id: str | None = Field(None, max_length=120)  # mã bản ghi bên Nexagri, để gộp bản cập nhật


# Khoảng hợp lý (rộng) để bắt lỗi cảm biến/nhập liệu; chuyên gia chỉnh theo từng ngành.
RANGES: dict[str, tuple[float, float]] = {
    "temperature": (-10, 60), "water_temperature": (0, 45), "air_temperature": (-10, 60),
    "ph": (0, 14), "do": (0, 30), "salinity": (0, 60), "ammonia": (0, 50), "nh3": (0, 50),
    "humidity": (0, 100), "soil_moisture": (0, 100), "rainfall": (0, 1000), "ec": (0, 50),
    "feed_kg": (0, 100000), "weight_kg": (0, 5000), "eggs": (0, 1_000_000),
}


class Quality(str, Enum):
    RAW = "raw"
    CHECKED = "checked"


class StoredRecord(BaseModel):
    farm: str  # mã giả
    plot: str | None
    kind: RecordKind
    at: datetime
    sector: str | None
    species: str | None
    province: str | None
    lat: float | None
    lon: float | None
    values: dict[str, float]
    units: dict[str, str]
    note: str | None
    vision: VisionInfo | None = None
    source: str | None = None  # mã giả của source_id
    quality: Quality
    flags: list[str]
    ingested_at: datetime


class FarmDataLake:
    def __init__(self, root: Path, pseudonym_key: str) -> None:
        if len(pseudonym_key) < 32:
            raise ValueError("khoá mã giả phải dài ít nhất 32 ký tự")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = pseudonym_key.encode()
        self._lock = threading.Lock()

    def pseudonym(self, *parts: str) -> str:
        return hmac.new(self.key, "\x1f".join(parts).encode(), hashlib.sha256).hexdigest()[:24]

    @staticmethod
    def check(rec: FarmRecord, now: datetime) -> list[str]:
        flags = []
        if rec.kind == RecordKind.VISION_EVENT and rec.vision is None:
            flags.append("sự kiện camera thiếu thông tin AI")
        at = rec.at if rec.at.tzinfo else rec.at.replace(tzinfo=timezone.utc)
        if at > now + timedelta(days=1):
            flags.append("thời gian ở tương lai")
        for k, v in rec.values.items():
            if not math.isfinite(v):
                flags.append(f"{k}: không phải số hợp lệ")
                continue
            lo_hi = RANGES.get(k.lower())
            if lo_hi and not (lo_hi[0] <= v <= lo_hi[1]):
                flags.append(f"{k}={v} ngoài khoảng {lo_hi[0]}–{lo_hi[1]}")
        return flags

    def deidentify(self, rec: FarmRecord, now: datetime) -> StoredRecord:
        flags = self.check(rec, now)
        values = {k: v for k, v in rec.values.items() if math.isfinite(v)}
        return StoredRecord(
            farm=self.pseudonym(rec.tenant_id),
            plot=self.pseudonym(rec.tenant_id, rec.plot_id) if rec.plot_id else None,
            kind=rec.kind, at=rec.at, sector=rec.sector, species=rec.species, province=rec.province,
            lat=round(rec.lat, 1) if rec.lat is not None else None,
            lon=round(rec.lon, 1) if rec.lon is not None else None,
            values=values, units=rec.units, note=redact(rec.note) if rec.note else None,
            vision=rec.vision.model_copy(update={"camera": self.pseudonym(rec.tenant_id, "cam", rec.vision.camera)})
            if rec.vision and rec.vision.camera else rec.vision,
            source=self.pseudonym(rec.tenant_id, "src", rec.source_id) if rec.source_id else None,
            quality=Quality.RAW if flags else Quality.CHECKED, flags=flags, ingested_at=now,
        )

    def write(self, recs: list[FarmRecord]) -> tuple[int, int]:
        """Ghi các bản ghi (đã được kiểm tra đồng ý ở lớp gọi). Trả (số ghi, số gắn cờ)."""
        now = datetime.now(timezone.utc)
        flagged = 0
        groups: dict[Path, list[str]] = {}
        for rec in recs:
            s = self.deidentify(rec, now)
            flagged += bool(s.flags)
            path = self.root / s.farm / f"{s.at:%Y-%m}.jsonl"
            groups.setdefault(path, []).append(s.model_dump_json())
        with self._lock:
            for path, lines in groups.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
        return len(recs), flagged

    def read(self, tenant_id: str) -> list[StoredRecord]:
        d = self.root / self.pseudonym(tenant_id)
        out: list[StoredRecord] = []
        for p in sorted(d.glob("*.jsonl")) if d.is_dir() else []:
            out.extend(StoredRecord.model_validate(json.loads(x)) for x in p.read_text(encoding="utf-8").splitlines() if x)
        return out

    def erase(self, tenant_id: str) -> bool:
        d = self.root / self.pseudonym(tenant_id)
        with self._lock:
            if d.is_dir():
                shutil.rmtree(d)
                return True
        return False
