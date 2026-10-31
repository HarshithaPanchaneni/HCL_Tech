"""Load, clean, translate, and embed the IT ticket dataset into ChromaDB.

Primary source: data/raw/Dataset_ticket.xlsx (Category, Subcategory, Text — Italian)
Fallback: HuggingFace Noise144/ticket_classification_IT_EN
18 subcategories, 100 tickets each, 1800 total.
Tickets are translated from Italian to English using argos-translate (offline).

Usage: python -m src.data_pipeline
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL,
    TICKET_COLLECTION,
    VECTOR_DB_PATH,
    Ticket,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TICKETS_JSON = PROCESSED_DIR / "tickets.json"
XLSX_PATH = RAW_DIR / "Dataset_ticket.xlsx"

PRIORITY_MAP = {
    "Database": "P2",
    "Hardware": "P3",
    "Network": "P2",
    "Software": "P3",
    "Inquiry/Help": "P4",
}


def load_dataset_xlsx() -> list[dict]:
    """Load tickets from the local xlsx file."""
    import openpyxl
    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True)
    ws = wb.active
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row and len(row) >= 3 and row[2]:
            rows.append({
                "Category": str(row[0] or "").strip(),
                "Subcategory": str(row[1] or "").strip(),
                "Text": str(row[2] or "").strip(),
            })
    wb.close()
    return rows


def load_dataset_hf() -> list[dict]:
    """Fallback: download from HuggingFace."""
    from datasets import load_dataset as hf_load
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ds = hf_load("Noise144/ticket_classification_IT_EN", split="train")
    return [dict(row) for row in ds]


def download_dataset() -> list[dict]:
    """Load dataset from xlsx (preferred) or HuggingFace (fallback)."""
    if XLSX_PATH.is_file():
        print(f"Loading from {XLSX_PATH}...")
        return load_dataset_xlsx()
    print("xlsx not found, trying HuggingFace...")
    return load_dataset_hf()


def _extract_title(text: str) -> str:
    """Extract a title from the first sentence or first 100 chars."""
    text = text.strip()
    for sep in (". ", ".\n", "\n"):
        idx = text.find(sep)
        if 10 < idx < 120:
            return text[:idx]
    return text[:100].rsplit(" ", 1)[0] if len(text) > 100 else text


def clean_tickets(raw_rows: list[dict]) -> list[dict]:
    """Clean and normalize raw dataset rows into Ticket-compatible dicts."""
    seen: set[str] = set()
    tickets: list[dict] = []

    for row in raw_rows:
        # Dataset columns: Category, Subcategory, Text
        text = str(row.get("Text", "") or row.get("text", "") or row.get("body", "") or row.get("description", "") or "").strip()
        category = str(row.get("Category", "") or row.get("category", "") or row.get("ticket_type", "") or "").strip()
        subcategory = str(row.get("Subcategory", "") or row.get("subcategory", "") or row.get("sub_category", "") or "").strip()

        if not text:
            continue

        dedup_key = hashlib.md5(text.lower().encode()).hexdigest()
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        title = _extract_title(text)
        priority = PRIORITY_MAP.get(category, "P3")
        ticket_id = f"TKT-{len(tickets)+1:04d}"

        tickets.append({
            "id": ticket_id,
            "title": title,
            "description": text,
            "category": category,
            "subcategory": subcategory,
            "priority": priority,
            "status": "closed",
            "resolution": None,
            "metadata": {
                "source": "huggingface",
                "dataset": "Noise144/ticket_classification_IT_EN",
                "language": "it",
            },
        })

    return tickets


def translate_tickets(tickets: list[dict]) -> list[dict]:
    """Translate Italian tickets to English using argos-translate (offline)."""
    import argostranslate.translate
    import argostranslate.package

    installed = argostranslate.package.get_installed_packages()
    has_it_en = any(p.from_code == "it" and p.to_code == "en" for p in installed)
    if not has_it_en:
        argostranslate.package.update_package_index()
        pkgs = argostranslate.package.get_available_packages()
        it_en = next(p for p in pkgs if p.from_code == "it" and p.to_code == "en")
        argostranslate.package.install_from_path(it_en.download())

    for i, t in enumerate(tickets):
        if t.get("metadata", {}).get("language") == "it":
            original = t["description"]
            translated = argostranslate.translate.translate(original, "it", "en")
            t["title"] = _extract_title(translated)
            t["description"] = translated
            t["metadata"]["language"] = "en"
            t["metadata"]["original_it"] = original
        if (i + 1) % 200 == 0:
            print(f"  Translated {i+1}/{len(tickets)}...")

    return tickets


def save_tickets(tickets: list[dict]) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TICKETS_JSON.write_text(json.dumps(tickets, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return TICKETS_JSON


def embed_tickets(tickets: list[dict], batch_size: int = 64) -> int:
    """Embed tickets and upsert into ChromaDB."""
    db_path = PROJECT_ROOT / VECTOR_DB_PATH
    db_path.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        name=TICKET_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Structured embedding: include category context for better retrieval
    texts = [
        f"Category: {t['category']}. Subcategory: {t['subcategory']}. {t['description']}"
        for t in tickets
    ]

    embeddings = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True
    ).tolist()

    # ChromaDB batch limit — upsert in chunks
    chunk_size = 500
    for start in range(0, len(tickets), chunk_size):
        end = start + chunk_size
        chunk = tickets[start:end]
        collection.upsert(
            ids=[t["id"] for t in chunk],
            embeddings=embeddings[start:end],
            documents=[
                f"{t['title']}. {t['description']}" for t in chunk
            ],
            metadatas=[
                {
                    "category": t["category"] or "",
                    "subcategory": t["subcategory"] or "",
                    "priority": t["priority"] or "P3",
                    "has_resolution": False,
                }
                for t in chunk
            ],
        )

    return collection.count()


def main() -> None:
    print("Downloading IT ticket dataset from HuggingFace...")
    raw = download_dataset()
    print(f"Downloaded {len(raw)} raw rows.")

    print("Cleaning and normalizing tickets...")
    tickets = clean_tickets(raw)
    print(f"Cleaned to {len(tickets)} unique tickets.")

    print("Translating Italian tickets to English...")
    tickets = translate_tickets(tickets)
    print("Translation complete.")

    path = save_tickets(tickets)
    print(f"Saved to {path}")

    print(f"Embedding tickets into ChromaDB using {EMBEDDING_MODEL}...")
    count = embed_tickets(tickets)
    print(f"Indexed {count} tickets into ChromaDB collection '{TICKET_COLLECTION}'.")
    print(f"Database path: {PROJECT_ROOT / VECTOR_DB_PATH}")


if __name__ == "__main__":
    main()
