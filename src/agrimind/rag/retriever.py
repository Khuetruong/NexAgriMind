"""Tìm đoạn tài liệu liên quan.

Bản đầu dùng BM25 (chạy không cần GPU, không cần dịch vụ ngoài).
Dùng công thức IDF kiểu Lucene (luôn dương) để chạy đúng cả khi kho còn nhỏ.
Khi có Qdrant + mô hình embedding, thay bằng lớp cùng giao diện `search`.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from ..corpus.models import Chunk

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    # Giữ dấu tiếng Việt (dấu phân biệt nghĩa), chỉ chuẩn hóa NFC + chữ thường.
    text = unicodedata.normalize("NFC", text).lower()
    words = _WORD.findall(text)
    # Thêm cặp từ để bắt từ ghép tiếng Việt ("chất lượng", "thức ăn").
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


@dataclass
class Hit:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("kho tài liệu rỗng")
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(tokenize(c.title + " " + c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / len(self._len)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def _score(self, i: int, q: list[str]) -> float:
        tf, dl = self._tf[i], self._len[i]
        s = 0.0
        for t in q:
            f = tf.get(t)
            if f:
                s += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avg))
        return s

    def search(self, query: str, k: int = 5, domains: list[str] | None = None) -> list[Hit]:
        q = tokenize(query)
        if not q:
            return []
        hits = []
        for i, c in enumerate(self.chunks):
            if domains and not set(domains) & set(c.domains):
                continue
            sc = self._score(i, q)
            if sc > 0:
                hits.append(Hit(c, sc))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
