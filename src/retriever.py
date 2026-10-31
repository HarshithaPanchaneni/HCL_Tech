"""Dual retriever: similar tickets + hybrid KB retrieval (fast path + vector fallback).

The ticket corpus is Italian; the KB is English. Both use a multilingual embedding model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    KB_COLLECTION,
    KB_MAPPING_PATH,
    TICKET_COLLECTION,
    VECTOR_DB_PATH,
    ClassificationResult,
    KnowledgeArticle,
    SimilarTicket,
    Ticket,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        for key in (name, name.capitalize(), name.upper()):
            if key in obj:
                return obj[key]
        return default
    return getattr(obj, name, default)


def _first_nonempty(*values: Any) -> str:
    for v in values:
        if v is not None:
            text = str(v).strip()
            if text:
                return text
    return ""


def _cosine_similarity(a: Any, b: Any) -> float:
    a_vec = np.asarray(a, dtype=np.float32).reshape(-1)
    b_vec = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


def _load_subcategory_mapping(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for item in payload.get("mappings", []):
        key = (str(item["category"]).strip().casefold(), str(item["subcategory"]).strip().casefold())
        mapping[key] = item
    return mapping


class DualRetriever:
    def __init__(self, db_path: str | Path | None = None) -> None:
        db_path = Path(db_path) if db_path else PROJECT_ROOT / VECTOR_DB_PATH
        if not db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB path does not exist: {db_path}. "
                "Run 'python -m src.data_pipeline' and 'python -m src.kb_indexer' first."
            )

        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._mapping = _load_subcategory_mapping(PROJECT_ROOT / KB_MAPPING_PATH)

        try:
            self._tickets = self._client.get_collection(name=TICKET_COLLECTION)
        except Exception:
            self._tickets = None

        try:
            self._kb = self._client.get_collection(name=KB_COLLECTION)
        except Exception:
            self._kb = None

    def _embed(self, text: str) -> list[float]:
        return np.asarray(
            self._model.encode(text, convert_to_numpy=True), dtype=np.float32
        ).reshape(-1).tolist()

    # ── Ticket Similarity ──

    def find_similar_tickets(
        self,
        ticket: Ticket | dict,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> list[SimilarTicket]:
        if self._tickets is None or self._tickets.count() == 0:
            return []

        title = _first_nonempty(_field(ticket, "title"), _field(ticket, "Title"))
        description = _first_nonempty(
            _field(ticket, "description"), _field(ticket, "text"), _field(ticket, "Text")
        )
        query_text = ". ".join(p for p in (title, description) if p)
        if not query_text:
            return []

        embedding = self._embed(query_text)

        where_filter = {"category": category_filter} if category_filter else None

        results = self._tickets.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._tickets.count()),
            where=where_filter,
            include=["documents", "distances", "metadatas"],
        )

        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]

        similar: list[SimilarTicket] = []
        for doc, dist, meta, tid in zip(documents, distances, metadatas, ids):
            score = 1.0 - float(dist)
            parts = doc.split(". ", 1) if doc else ["", ""]
            t = Ticket(
                id=tid,
                title=parts[0] if parts else "",
                description=parts[1] if len(parts) > 1 else doc or "",
                category=meta.get("category"),
                subcategory=meta.get("subcategory"),
                priority=meta.get("priority"),
                status="closed",
                resolution=None,
                metadata={"has_resolution": meta.get("has_resolution", False)},
            )
            similar.append(SimilarTicket(
                ticket=t,
                similarity_score=score,
                matching_reason=f"{meta.get('category', '?')}/{meta.get('subcategory', '?')}, {score:.0%} similarity",
            ))

        return similar

    # ── Knowledge Base — Hybrid Retrieval ──

    def _kb_article_from_doc(self, document: str, score: float, source: str) -> KnowledgeArticle:
        data = json.loads(document)
        return KnowledgeArticle(
            kb_id=data["kb_id"],
            title=data["title"],
            category=data["category"],
            subcategory=data["subcategory"],
            symptoms=data["symptoms"],
            resolution_steps=data["resolution"]["steps"],
            diagnostic_steps=data["diagnostic_steps"],
            escalation_criteria=data["escalation_criteria"],
            relevance_score=score,
            retrieval_source=source,
        )

    def _mapped_candidates(
        self, candidate_ids: list[str], query_embedding: list[float]
    ) -> list[KnowledgeArticle]:
        if not candidate_ids or self._kb is None:
            return []
        records = self._kb.get(ids=candidate_ids, include=["documents", "embeddings"])
        documents = records.get("documents") or []
        embeddings = records.get("embeddings")
        if embeddings is None:
            return []

        results: list[KnowledgeArticle] = []
        for doc, emb in zip(documents, embeddings):
            if doc:
                score = _cosine_similarity(query_embedding, emb)
                results.append(self._kb_article_from_doc(doc, score, "subcategory_map"))
        return sorted(results, key=lambda a: a.relevance_score, reverse=True)

    def _vector_candidates(
        self, query_embedding: list[float], top_k: int, where_filter: dict | None
    ) -> list[KnowledgeArticle]:
        if self._kb is None or self._kb.count() == 0:
            return []
        results = self._kb.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._kb.count()),
            where=where_filter,
            include=["documents", "distances"],
        )
        documents = (results.get("documents") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        return [
            self._kb_article_from_doc(doc, 1.0 - float(dist), "vector_fallback")
            for doc, dist in zip(documents, distances)
            if doc
        ]

    def find_knowledge_articles(
        self,
        ticket: Ticket | dict,
        classification: ClassificationResult | None = None,
        top_k: int = 3,
    ) -> list[KnowledgeArticle]:
        if self._kb is None or self._kb.count() == 0:
            return []

        title = _first_nonempty(_field(ticket, "title"), _field(ticket, "Title"))
        description = _first_nonempty(
            _field(ticket, "description"), _field(ticket, "text"), _field(ticket, "Text")
        )
        category = _first_nonempty(
            _field(classification, "category"),
            _field(ticket, "category"), _field(ticket, "Category"),
        )
        subcategory = _first_nonempty(
            _field(classification, "subcategory"),
            _field(ticket, "subcategory"), _field(ticket, "Subcategory"),
        )

        query_parts = []
        if category:
            query_parts.append(f"Category: {category}")
        if subcategory:
            query_parts.append(f"Subcategory: {subcategory}")
        if title:
            query_parts.append(f"Title: {title}")
        if description:
            query_parts.append(f"Ticket text: {description}")
        query_text = ". ".join(query_parts)
        if not query_text:
            return []

        query_embedding = self._embed(query_text)

        # Fast path: subcategory mapping lookup
        mapping_key = (category.casefold(), subcategory.casefold()) if category and subcategory else None
        mapping_entry = self._mapping.get(mapping_key)
        candidate_ids = mapping_entry.get("kb_ids", []) if mapping_entry else []
        coverage = mapping_entry.get("coverage", "unsupported") if mapping_entry else "unsupported"

        mapped = self._mapped_candidates(candidate_ids, query_embedding)

        # Decide if we need vector fallback
        needs_vector = (
            not mapped
            or coverage in {"partial", "unsupported"}
            or mapped[0].relevance_score < 0.40
        )

        # Category filter for vector search
        confidence = float(_field(classification, "confidence", 0.0) or 0.0)
        has_dataset_labels = bool(
            _first_nonempty(_field(ticket, "category"), _field(ticket, "Category"))
            and _first_nonempty(_field(ticket, "subcategory"), _field(ticket, "Subcategory"))
        )
        # Map dataset category to KB category for filtering
        category_for_filter = mapping_entry.get("category") if mapping_entry else category
        where_filter = (
            {"category": category_for_filter}
            if category_for_filter and (confidence > 0.8 or has_dataset_labels)
            else None
        )

        vector = self._vector_candidates(query_embedding, top_k, where_filter) if needs_vector else []

        # Also check secondary classification
        if classification:
            sec_cat = _field(classification, "secondary_category")
            sec_sub = _field(classification, "secondary_subcategory")
            sec_conf = float(_field(classification, "secondary_confidence", 0.0) or 0.0)
            if sec_cat and sec_sub and sec_conf > 0.4:
                sec_key = (str(sec_cat).casefold(), str(sec_sub).casefold())
                sec_entry = self._mapping.get(sec_key)
                sec_ids = sec_entry.get("kb_ids", []) if sec_entry else []
                mapped.extend(self._mapped_candidates(sec_ids, query_embedding))

        # Merge by kb_id, keep highest score
        merged: dict[str, KnowledgeArticle] = {}
        for article in mapped + vector:
            current = merged.get(article.kb_id)
            if current is None or article.relevance_score > current.relevance_score:
                merged[article.kb_id] = article

        ranked = sorted(merged.values(), key=lambda a: a.relevance_score, reverse=True)
        return ranked[:top_k]


_default_retriever: DualRetriever | None = None


def _get_retriever() -> DualRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = DualRetriever()
    return _default_retriever


def find_similar_tickets(
    ticket: Ticket | dict,
    top_k: int = 5,
    category_filter: str | None = None,
) -> list[SimilarTicket]:
    return _get_retriever().find_similar_tickets(ticket, top_k, category_filter)


def find_knowledge_articles(
    ticket: Ticket | dict,
    classification: ClassificationResult | None = None,
    top_k: int = 3,
) -> list[KnowledgeArticle]:
    return _get_retriever().find_knowledge_articles(ticket, classification, top_k)
