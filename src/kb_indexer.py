"""Validate and index the IT support knowledge base into ChromaDB.

Usage: python -m src.kb_indexer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    KB_COLLECTION,
    KB_MAPPING_PATH,
    KB_PATH,
    VECTOR_DB_PATH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = {
    "kb_id", "title", "category", "subcategory", "symptoms",
    "common_error_codes", "root_cause", "diagnostic_steps", "resolution",
    "prevention", "escalation_criteria", "related_kb_ids", "tags",
    "priority_hint", "last_updated",
}


def load_knowledge_base(path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path else PROJECT_ROOT / KB_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge base JSON not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        articles = json.load(f)

    if not isinstance(articles, list):
        raise ValueError("Knowledge base must be a JSON array.")

    seen_ids: set[str] = set()
    for i, article in enumerate(articles):
        missing = REQUIRED_KEYS - article.keys()
        if missing:
            raise ValueError(f"Article at index {i} missing keys: {sorted(missing)}")
        kb_id = article["kb_id"]
        if kb_id in seen_ids:
            raise ValueError(f"Duplicate knowledge-base ID: {kb_id}")
        seen_ids.add(kb_id)

    return articles


def _validate_subcategory_mapping(known_ids: set[str]) -> None:
    mapping_path = PROJECT_ROOT / KB_MAPPING_PATH
    if not mapping_path.is_file():
        print(f"Warning: subcategory mapping not found at {mapping_path}, skipping validation.")
        return

    with mapping_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    for item in payload.get("mappings", []):
        for kb_id in item.get("kb_ids", []):
            if kb_id not in known_ids:
                raise ValueError(
                    f"Mapping ({item['category']}, {item['subcategory']}) "
                    f"references unknown KB ID: {kb_id}"
                )
    print(f"Subcategory mapping validated ({len(payload.get('mappings', []))} entries).")


def _article_to_embedding_text(article: dict[str, Any]) -> str:
    resolution_steps = " ".join(article["resolution"]["steps"])
    return (
        f"{article['title']}. Category: {article['category']}. "
        f"Subcategory: {article['subcategory']}. "
        f"Symptoms: {' '.join(article['symptoms'])}. "
        f"Common errors: {' '.join(article['common_error_codes'])}. "
        f"Root cause: {article['root_cause']}. "
        f"Resolution: {resolution_steps}. "
        f"Tags: {' '.join(article['tags'])}."
    )


def index_knowledge_base(articles: list[dict[str, Any]] | None = None) -> int:
    if articles is None:
        articles = load_knowledge_base()

    known_ids = {a["kb_id"] for a in articles}
    _validate_subcategory_mapping(known_ids)

    db_path = PROJECT_ROOT / VECTOR_DB_PATH
    db_path.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=KB_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [_article_to_embedding_text(a) for a in articles]
    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True).tolist()

    collection.upsert(
        ids=[a["kb_id"] for a in articles],
        embeddings=embeddings,
        documents=[json.dumps(a, ensure_ascii=False) for a in articles],
        metadatas=[
            {
                "category": a["category"],
                "subcategory": a["subcategory"],
                "priority_hint": a["priority_hint"],
                "requires_admin": a["resolution"]["requires_admin"],
                "title": a["title"],
                "tags": ",".join(a["tags"]),
            }
            for a in articles
        ],
    )

    return collection.count()


def main() -> None:
    count = index_knowledge_base()
    print(f"Indexed {count} KB articles into ChromaDB collection '{KB_COLLECTION}'.")
    print(f"Database path: {PROJECT_ROOT / VECTOR_DB_PATH}")


if __name__ == "__main__":
    main()
