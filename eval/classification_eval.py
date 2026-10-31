"""Classification evaluation script.

Evaluates ticket classification performance against ground truth labels.
Saves metrics to eval/results/classification_metrics.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import classify_ticket, classify_ticket_keyword_fallback
from src.config import Ticket
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
SAMPLE_TICKETS_PATH = PROJECT_ROOT / "data" / "sample_tickets.json"
PROCESSED_TICKETS_PATH = PROJECT_ROOT / "data" / "processed" / "tickets.json"


def evaluate_samples(sample_tickets: list[dict]) -> dict:
    total = 0
    correct_category = 0
    correct_subcategory = 0
    results = []

    for st in sample_tickets:
        expected_cat = st.get("metadata", {}).get("expected_category")
        expected_subcat = st.get("metadata", {}).get("expected_subcategory")
        if not expected_cat:
            continue

        ticket = Ticket(id=st["id"], title=st["title"], description=st["description"])
        pred = classify_ticket(ticket)

        cat_match = pred.category.lower() == expected_cat.lower()
        subcat_match = pred.subcategory.lower() == expected_subcat.lower() if expected_subcat else False

        total += 1
        if cat_match:
            correct_category += 1
        if subcat_match:
            correct_subcategory += 1

        results.append({
            "id": st["id"],
            "title": st["title"],
            "expected_category": expected_cat,
            "expected_subcategory": expected_subcat,
            "predicted_category": pred.category,
            "predicted_subcategory": pred.subcategory,
            "predicted_priority": pred.priority,
            "confidence": round(pred.confidence, 4),
            "category_match": cat_match,
            "subcategory_match": subcat_match,
        })

    cat_acc = correct_category / total if total else 0.0
    subcat_acc = correct_subcategory / total if total else 0.0

    return {
        "num_evaluated": total,
        "category_accuracy": round(cat_acc, 4),
        "subcategory_accuracy": round(subcat_acc, 4),
        "details": results,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_TICKETS_PATH, encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} sample tickets for classification evaluation...")
    metrics = evaluate_samples(samples)

    print(f"\nClassification Results:")
    print(f"  Category Accuracy:    {metrics['category_accuracy']:.1%}")
    print(f"  Subcategory Accuracy: {metrics['subcategory_accuracy']:.1%}")

    out_file = RESULTS_DIR / "classification_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out_file}")


if __name__ == "__main__":
    main()
