"""Nạp tài liệu mở từ danh sách nguồn: giấy phép, lấy chữ, chạy lại không ghi đè."""

from pathlib import Path

import httpx

from agrimind.corpus.ingest import build_chunks, load_directory
from agrimind.corpus.models import License
from agrimind.corpus.sources import Fetcher, fetch_manifest, html_to_text, normalize_doi, pdf_to_text

LONG = "Oxy hoa tan thap vao ban dem lam tom giam an. " * 10


def _pdf(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offs:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)


class Web:
    def __init__(self):
        self.pages = {
            "/paper.pdf": ("application/pdf", _pdf(LONG)),
            "/guide": ("text/html", f"<html><script>x()</script><nav>menu</nav><p>{LONG}</p></html>".encode()),
            "/short": ("text/html", b"<html><p>Dang nhap</p></html>"),
            "/nc": ("text/plain", LONG.encode()),
        }
        self.works = {
            "10.1/oa": {"title": "Bai bao mo", "publication_year": 2021, "language": "en",
                        "best_oa_location": {"license": "cc-by", "pdf_url": "https://web/paper.pdf"}},
            "10.1/nc": {"title": "Bai bao NC", "publication_year": 2020,
                        "best_oa_location": {"license": "cc-by-nc", "landing_page_url": "https://web/nc"}},
        }

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if req.url.host == "oa":
            doi = req.url.path.split("doi:", 1)[1]
            w = self.works.get(doi)
            return httpx.Response(200, json=w) if w else httpx.Response(404)
        ctype, body = self.pages.get(req.url.path, ("", b""))
        return httpx.Response(200, content=body, headers={"content-type": ctype}) if body else httpx.Response(404)


def _fetcher(web=None):
    return Fetcher(httpx.Client(transport=httpx.MockTransport(web or Web())), openalex="https://oa")


def test_extractors():
    assert "tom giam an" in pdf_to_text(_pdf(LONG))
    t = html_to_text("<script>bad()</script><p>Một</p><p>Hai &amp; ba</p>")
    assert "bad" not in t and "Hai & ba" in t
    assert normalize_doi("https://doi.org/10.1/AB") == "10.1/ab"


def test_fetch_manifest_licences_and_rerun(tmp_path):
    m = tmp_path / "sources.yaml"
    m.write_text("""
sources:
  - {id: bai-mo, doi: "10.1/oa", domains: [D02]}
  - {id: bai-nc, doi: "doi:10.1/nc", domains: [D02]}
  - {id: huong-dan, url: "https://web/guide", title: "Huong dan", license: gov-public, domains: [D01], language: vi}
  - {id: xung-dot, doi: "10.1/nc", license: CC-BY}
  - {id: qua-ngan, url: "https://web/short", title: "x", license: CC0}
  - {id: khong-ro, url: "https://web/nc", title: "Khong ro"}
""", encoding="utf-8")
    out = tmp_path / "corpus"
    rep = fetch_manifest(m, out, _fetcher())
    assert sorted(rep.written) == ["bai-mo", "bai-nc", "huong-dan", "khong-ro"]
    assert sorted(rep.unknown_license) == ["bai-nc", "khong-ro"]
    assert "OpenAlex" in rep.errors["xung-dot"] and "quá ít chữ" in rep.errors["qua-ngan"]

    docs = {d.id: d for d in load_directory(out)}
    assert docs["bai-mo"].license == License.CC_BY and docs["bai-mo"].license_source == "openalex"
    assert docs["bai-mo"].year == 2021 and docs["bai-mo"].doi == "10.1/oa"
    assert docs["huong-dan"].license == License.GOV_PUBLIC and "menu" not in docs["huong-dan"].text
    assert docs["khong-ro"].license == License.UNKNOWN
    served = {c.doc_id for c in build_chunks(list(docs.values()))}
    assert served == {"bai-mo", "huong-dan"}

    rep2 = fetch_manifest(m, out, _fetcher())
    assert sorted(rep2.unchanged) == ["bai-mo", "bai-nc", "huong-dan", "khong-ro"] and not rep2.written
