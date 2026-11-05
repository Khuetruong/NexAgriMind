"""Nạp tài liệu mở từ danh sách nguồn (gói 2 của A10).

Chuyên gia chọn nguồn và ghi vào tệp YAML; công cụ này tải về, lấy chữ, ghi tệp
.md có phần đầu YAML vào kho. Quy tắc giấy phép (A04):
- Có DOI → tra giấy phép trên OpenAlex. Danh sách ghi giấy phép khác OpenAlex
  thì DỪNG tài liệu đó, chờ người kiểm tra (không tự chọn bên dễ dãi hơn).
- Không rõ giấy phép → vẫn lưu với `license: unknown`, bị loại khỏi kho sản phẩm.
- Ghi mã băm nội dung gốc; chạy lại mà nguồn không đổi thì bỏ qua.

Ví dụ tệp nguồn:

    sources:
      - id: vd-tom-oxy-2021
        doi: 10.xxxx/yyyy          # có DOI thì tự tra giấy phép, tiêu đề, năm
        domains: [D02]
      - id: tcvn-xxxx
        url: https://.../tieu-chuan.pdf
        title: "TCVN ..."
        license: gov-public
        domains: [D01, D04]
        language: vi
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel, Field

from .models import License

USER_AGENT = "AgriMindCorpus/0.1 (+tài liệu khoa học mở)"
MAX_BYTES = 30 * 1024 * 1024

_OPENALEX_LICENSE = {
    "cc0": License.CC0, "public-domain": License.PUBLIC_DOMAIN, "cc-by": License.CC_BY,
    "cc-by-sa": License.CC_BY_SA, "cc-by-nc": License.CC_BY_NC, "cc-by-nc-sa": License.CC_BY_NC_SA,
    "cc-by-nd": License.CC_BY_ND, "cc-by-nc-nd": License.CC_BY_NC_ND,
}


class Source(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,80}$")
    url: str | None = None
    doi: str | None = None
    title: str | None = None
    license: License | None = None
    domains: list[str] = Field(default_factory=list)
    year: int | None = None
    language: str = "en"


@dataclass
class Report:
    written: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    unknown_license: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


class _Text(HTMLParser):
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg", "form", "aside"}
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    p = _Text()
    p.feed(html)
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def pdf_to_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader  # giấy phép BSD; cài bằng pip install 'agrimind[pdf]'
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("cần gói pypdf để đọc PDF: pip install 'agrimind[pdf]'") from e
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()


def normalize_doi(doi: str) -> str:
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi.strip(), flags=re.I).lower()


class Fetcher:
    def __init__(self, client: httpx.Client | None = None, openalex: str = "https://api.openalex.org") -> None:
        self.http = client or httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        self.openalex = openalex.rstrip("/")

    def lookup_doi(self, doi: str) -> dict:
        r = self.http.get(f"{self.openalex}/works/doi:{normalize_doi(doi)}")
        r.raise_for_status()
        w = r.json()
        loc = w.get("best_oa_location") or {}
        lic = _OPENALEX_LICENSE.get((loc.get("license") or "").lower(), License.UNKNOWN)
        return {
            "title": w.get("title") or w.get("display_name"),
            "year": w.get("publication_year"),
            "language": w.get("language"),
            "license": lic,
            "url": loc.get("pdf_url") or loc.get("landing_page_url"),
        }

    def download(self, url: str) -> tuple[bytes, str]:
        with self.http.stream("GET", url) as r:
            r.raise_for_status()
            ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
            buf = bytearray()
            for part in r.iter_bytes():
                buf.extend(part)
                if len(buf) > MAX_BYTES:
                    raise ValueError("tệp quá lớn (>30MB)")
        return bytes(buf), ctype


def extract(data: bytes, ctype: str, url: str) -> str:
    if ctype == "application/pdf" or data[:5] == b"%PDF-" or url.lower().endswith(".pdf"):
        return pdf_to_text(data)
    text = data.decode("utf-8", errors="replace")
    if "html" in ctype or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        return html_to_text(text)
    return text.strip()


def _existing_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    m = re.search(r"^sha256:\s*([0-9a-f]{64})\s*$", path.read_text(encoding="utf-8")[:4000], re.M)
    return m.group(1) if m else None


def fetch_manifest(manifest: Path, out_dir: Path, fetcher: Fetcher | None = None) -> Report:
    fetcher = fetcher or Fetcher()
    data = yaml.safe_load(Path(manifest).read_text(encoding="utf-8")) or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Report()
    seen: set[str] = set()
    for raw in data.get("sources", []):
        sid = str((raw or {}).get("id", "?"))
        try:
            src = Source(**raw)
            if src.id in seen:
                raise ValueError("trùng id")
            seen.add(src.id)
            meta = {"title": src.title, "year": src.year, "language": src.language, "url": src.url}
            license, license_source = src.license, "manifest" if src.license else None
            if src.doi:
                oa = fetcher.lookup_doi(src.doi)
                for k in ("title", "year", "language", "url"):
                    meta[k] = meta[k] or oa[k]
                if license and oa["license"] != License.UNKNOWN and oa["license"] != license:
                    raise ValueError(f"giấy phép trong danh sách ({license.value}) khác OpenAlex "
                                     f"({oa['license'].value}); cần người kiểm tra")
                if not license:
                    license, license_source = oa["license"], "openalex"
            license = license or License.UNKNOWN
            if not meta["url"]:
                raise ValueError("không có url để tải")
            if not meta["title"]:
                raise ValueError("thiếu tiêu đề")
            body, ctype = fetcher.download(meta["url"])
            sha = hashlib.sha256(body).hexdigest()
            path = out_dir / f"{src.id}.md"
            if _existing_sha(path) == sha:
                rep.unchanged.append(src.id)
                continue
            text = extract(body, ctype, meta["url"])
            if len(text) < 200:
                raise ValueError("lấy được quá ít chữ (có thể là trang đăng nhập hoặc ảnh quét)")
            front = {
                "id": src.id, "title": meta["title"], "source": meta["url"], "license": license.value,
                "license_source": license_source or "none", "doi": normalize_doi(src.doi) if src.doi else None,
                "domains": src.domains, "year": meta["year"], "language": meta["language"] or "en",
                "quality": "raw", "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sha256": sha,
            }
            front = {k: v for k, v in front.items() if v is not None}
            path.write_text("---\n" + yaml.safe_dump(front, allow_unicode=True, sort_keys=False) + "---\n"
                            + text + "\n", encoding="utf-8")
            rep.written.append(src.id)
            if not license.commercial_ok:
                rep.unknown_license.append(src.id)
        except (httpx.HTTPError, ValueError, RuntimeError) as e:
            rep.errors[sid] = str(e)
    return rep
