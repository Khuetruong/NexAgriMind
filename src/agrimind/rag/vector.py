"""Tìm theo nghĩa (vector) bằng Qdrant + mô hình embedding, và gộp với BM25.

- Embedding gọi qua API `/embeddings` tương thích OpenAI (vLLM hoặc
  text-embeddings-inference chạy bge-m3, giấy phép MIT – A04).
- Qdrant gọi qua REST bằng httpx, không cần thư viện riêng.
- `HybridRetriever` gộp thứ hạng BM25 và vector bằng RRF (Reciprocal Rank Fusion):
  từ khoá chính xác (tên bệnh, tên hoạt chất) và câu hỏi diễn đạt khác đều tìm được.

Đoạn chỉ được coi là "đủ liên quan" khi ít nhất một bên vượt ngưỡng của chính nó;
trợ lý vẫn từ chối khi không có đoạn nào đủ liên quan.
"""

from __future__ import annotations

import uuid
from typing import Protocol

import httpx

from ..corpus.models import Chunk
from .retriever import BM25Retriever, Hit


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60,
                 client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = client or httpx.Client(timeout=timeout)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.http.post(f"{self.base_url}/embeddings", headers=self.headers,
                              json={"model": self.model, "input": texts})
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]


def point_id(chunk_id: str) -> str:
    """Qdrant chỉ nhận số hoặc UUID làm mã điểm; đổi mã đoạn thành UUID cố định."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "agrimind:" + chunk_id))


class QdrantStore:
    def __init__(self, url: str, collection: str = "agrimind_corpus", api_key: str | None = None,
                 timeout: float = 30, client: httpx.Client | None = None) -> None:
        self.url = url.rstrip("/")
        self.collection = collection
        self.headers = {"api-key": api_key} if api_key else {}
        self.http = client or httpx.Client(timeout=timeout)

    def _req(self, method: str, path: str, json: dict | None = None) -> dict:
        r = self.http.request(method, f"{self.url}/collections/{self.collection}{path}",
                              headers=self.headers, json=json)
        r.raise_for_status()
        return r.json()

    def ensure_collection(self, dim: int) -> None:
        r = self.http.get(f"{self.url}/collections/{self.collection}", headers=self.headers)
        if r.status_code == 200:
            return
        self._req("PUT", "", {"vectors": {"size": dim, "distance": "Cosine"}})

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("số đoạn và số vector không khớp")
        points = [
            {"id": point_id(c.id), "vector": v, "payload": c.model_dump(mode="json")}
            for c, v in zip(chunks, vectors)
        ]
        self._req("PUT", "/points?wait=true", {"points": points})

    def search(self, vector: list[float], k: int, domains: list[str] | None = None) -> list[tuple[Chunk, float]]:
        body: dict = {"vector": vector, "limit": k, "with_payload": True}
        if domains:
            body["filter"] = {"must": [{"key": "domains", "match": {"any": domains}}]}
        res = self._req("POST", "/points/search", body).get("result", [])
        out = []
        for p in res:
            chunk = Chunk(**p["payload"])
            # Lớp bảo vệ thứ hai: kho phục vụ chỉ được trả tài liệu có giấy phép thương mại.
            if chunk.license.commercial_ok:
                out.append((chunk, float(p["score"])))
        return out


class VectorRetriever:
    def __init__(self, store: QdrantStore, embedder: Embedder, min_similarity: float = 0.5) -> None:
        self.store, self.embedder, self.min_similarity = store, embedder, min_similarity

    def index(self, chunks: list[Chunk], batch: int = 64) -> int:
        n = 0
        for i in range(0, len(chunks), batch):
            part = [c for c in chunks[i:i + batch] if c.license.commercial_ok]
            if not part:
                continue
            vecs = self.embedder.embed([c.title + "\n" + c.text for c in part])
            if n == 0:
                self.store.ensure_collection(len(vecs[0]))
            self.store.upsert(part, vecs)
            n += len(part)
        return n

    def search(self, query: str, k: int = 5, domains: list[str] | None = None) -> list[Hit]:
        [vec] = self.embedder.embed([query])
        return [
            Hit(chunk, score, relevant=score >= self.min_similarity)
            for chunk, score in self.store.search(vec, k, domains)
        ]


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever, bm25_min: float = 1.0,
                 rrf_k: int = 60) -> None:
        self.bm25, self.vector, self.bm25_min, self.rrf_k = bm25, vector, bm25_min, rrf_k

    @property
    def chunks(self) -> list[Chunk]:
        return self.bm25.chunks

    def search(self, query: str, k: int = 5, domains: list[str] | None = None) -> list[Hit]:
        pool = max(k * 4, 20)
        lexical = self.bm25.search(query, k=pool, domains=domains)
        try:
            semantic = self.vector.search(query, k=pool, domains=domains)
        except httpx.HTTPError:
            # Qdrant hoặc embedding lỗi → vẫn trả lời bằng BM25, không sập dịch vụ.
            semantic = []
        fused: dict[str, Hit] = {}
        for ranked, relevant_of in (
            (lexical, lambda h: h.score >= self.bm25_min),
            (semantic, lambda h: bool(h.relevant)),
        ):
            for rank, h in enumerate(ranked):
                cur = fused.setdefault(h.chunk.id, Hit(h.chunk, 0.0, relevant=False))
                cur.score += 1.0 / (self.rrf_k + rank + 1)
                cur.relevant = bool(cur.relevant or relevant_of(h))
        out = sorted(fused.values(), key=lambda h: h.score, reverse=True)
        return out[:k]
