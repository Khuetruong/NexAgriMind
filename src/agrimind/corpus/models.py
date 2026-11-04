"""Tài liệu khoa học và đoạn trích (A03 – kho tài liệu, A04 – giấy phép).

Mỗi tài liệu bắt buộc ghi giấy phép. Tài liệu không cho dùng thương mại
vẫn được lưu để tham khảo nội bộ nhưng bị loại khỏi kho phục vụ sản phẩm.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class License(str, Enum):
    CC0 = "CC0"
    CC_BY = "CC-BY"
    CC_BY_SA = "CC-BY-SA"
    CC_BY_NC = "CC-BY-NC"
    CC_BY_NC_SA = "CC-BY-NC-SA"
    PUBLIC_DOMAIN = "public-domain"
    GOV_PUBLIC = "gov-public"  # văn bản pháp luật, tiêu chuẩn công bố công khai
    PARTNER = "partner"  # theo hợp đồng với đối tác
    PROPRIETARY = "proprietary"
    UNKNOWN = "unknown"

    @property
    def commercial_ok(self) -> bool:
        return self in {
            License.CC0,
            License.CC_BY,
            License.CC_BY_SA,
            License.PUBLIC_DOMAIN,
            License.GOV_PUBLIC,
            License.PARTNER,
        }


class Quality(str, Enum):
    """Bốn mức chất lượng dữ liệu của A03."""

    RAW = "raw"
    CHECKED = "checked"
    STANDARDIZED = "standardized"
    EXPERT_APPROVED = "expert_approved"


class Document(BaseModel):
    id: str
    title: str
    source: str = Field(description="URL hoặc trích dẫn đầy đủ")
    license: License
    language: str = "vi"
    domains: list[str] = Field(default_factory=list, description="Mã trung tâm D01..D10")
    year: int | None = None
    quality: Quality = Quality.RAW
    is_fixture: bool = Field(False, description="Dữ liệu mẫu cho kiểm thử, không phải tri thức thật")
    text: str


class Chunk(BaseModel):
    id: str
    doc_id: str
    title: str
    source: str
    license: License
    domains: list[str]
    text: str
