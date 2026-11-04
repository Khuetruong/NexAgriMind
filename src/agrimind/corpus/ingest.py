"""Nạp tài liệu Markdown/văn bản có phần đầu YAML và chia đoạn."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from .models import Chunk, Document

_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)


class IngestError(ValueError):
    pass


def parse_document(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    m = _FRONT.match(raw)
    if not m:
        raise IngestError(f"{path}: thiếu phần đầu YAML (id, title, source, license)")
    meta = yaml.safe_load(m.group(1)) or {}
    for key in ("id", "title", "source", "license"):
        if key not in meta:
            raise IngestError(f"{path}: thiếu trường bắt buộc '{key}'")
    return Document(**meta, text=raw[m.end():].strip())


def load_directory(root: Path) -> list[Document]:
    docs: list[Document] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("*.md")):
        doc = parse_document(path)
        if doc.id in seen:
            raise IngestError(f"{path}: trùng id '{doc.id}'")
        seen.add(doc.id)
        docs.append(doc)
    return docs


def chunk_document(doc: Document, max_chars: int = 1200) -> list[Chunk]:
    """Chia theo đoạn văn, gộp đoạn ngắn cho tới max_chars."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if buf:
            chunks.append(
                Chunk(
                    id=f"{doc.id}#{len(chunks)}",
                    doc_id=doc.id,
                    title=doc.title,
                    source=doc.source,
                    license=doc.license,
                    domains=doc.domains,
                    text="\n\n".join(buf),
                )
            )
        buf, size = [], 0

    for p in paras:
        if size and size + len(p) > max_chars:
            flush()
        buf.append(p)
        size += len(p)
    flush()
    return chunks


def build_chunks(docs: list[Document], commercial_only: bool = True) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        if commercial_only and not d.license.commercial_ok:
            continue
        out.extend(chunk_document(d))
    return out
