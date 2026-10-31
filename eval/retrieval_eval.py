"""Evaluate retrieval quality for both ticket similarity and KB article matching.

Usage: python -m eval.retrieval_eval
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
SAMPLE_TICKETS_PATH = PROJECT_ROOT / "data" / "sample_tickets.json"


def load_sample_tickets() -> list[dict]:
    with open(SAMPLE_TICKETS_PATH, encoding="utf-8") as f:
        return json.load(f)


def precision_at_k(relevant: list[bool], k: int) -> float:
    top = relevant[:k]
    if not top:
        return 0.0
    return sum(top) / len(top)


def mrr(relevant: list[bool]) -> float:
    for i, r in enumerate(relevant):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ticket_retrieval(sample_tickets: list[dict]) -> dict:
    from src.retriever import find_similar_tickets
    from src.config import Ticket

    results = []
    all_p1, all_p3, all_p5, all_mrr = [], [], [], []

    for st in sample_tickets:
        expected_cat = st.get("metadata", {}).get("expected_category")
        if not expected_cat:
            continue

        ticket = Ticket(id=st["id"], title=st["title"], description=st["description"])
        similar = find_similar_tickets(ticket, top_k=5)

        relevant = [s.ticket.category == expected_cat for s in similar]
        p1 = precision_at_k(relevant, 1)
        p3 = precision_at_k(relevant, 3)
        p5 = precision_at_k(relevant, 5)
        m = mrr(relevant)

        all_p1.append(p1)
        all_p3.append(p3)
        all_p5.append(p5)
        all_mrr.append(m)

        results.append({
            "ticket_id": st["id"],
            "expected_category": expected_cat,
            "top_5": [
                {
                    "id": s.ticket.id,
                    "category": s.ticket.category,
                    "subcategory": s.ticket.subcategory,
                    "score": round(s.similarity_score, 4),
                    "relevant": s.ticket.category == expected_cat,
                }
                for s in similar
            ],
            "precision@1": round(p1, 4),
            "precision@3": round(p3, 4),
            "precision@5": round(p5, 4),
            "mrr": round(m, 4),
        })

    n = len(all_p1) or 1
    return {
        "ticket_retrieval": {
            "mean_precision@1": round(sum(all_p1) / n, 4),
            "mean_precision@3": round(sum(all_p3) / n, 4),
            "mean_precision@5": round(sum(all_p5) / n, 4),
            "mean_mrr": round(sum(all_mrr) / n, 4),
            "num_queries": len(results),
            "per_query": results,
        }
    }


def evaluate_kb_retrieval(sample_tickets: list[dict]) -> dict:
    from src.retriever import find_knowledge_articles
    from src.config import Ticket, SUBCATEGORY_TO_KB

    results = []

    for st in sample_tickets:
        expected_cat = st.get("metadata", {}).get("expected_category")
        expected_subcat = st.get("metadata", {}).get("expected_subcategory")
        if not expected_cat or not expected_subcat:
            continue

        # Check which KB articles are mapped to this subcategory
        expected_kb_ids = set(SUBCATEGORY_TO_KB.get((expected_cat, expected_subcat), []))

        ticket = Ticket(id=st["id"], title=st["title"], description=st["description"])
        articles = find_knowledge_articles(ticket, top_k=3)

        # A match means the retrieved KB article is in the expected mapping
        matches = [a.kb_id in expected_kb_ids for a in articles] if expected_kb_ids else []
        # For unsupported subcategories, check category-level match on the KB article
        category_matches = [a.category == expected_cat or a.kb_id in expected_kb_ids for a in articles]

        results.append({
            "ticket_id": st["id"],
            "expected_category": expected_cat,
            "expected_subcategory": expected_subcat,
            "expected_kb_ids": sorted(expected_kb_ids),
            "has_mapping": bool(expected_kb_ids),
            "top_3": [
                {
                    "kb_id": a.kb_id,
                    "title": a.title,
                    "subcategory": a.subcategory,
                    "relevance_score": round(a.relevance_score, 4),
                    "retrieval_source": a.retrieval_source,
                    "in_expected_mapping": a.kb_id in expected_kb_ids,
                }
                for a in articles
            ],
            "any_mapped_match_in_top3": any(matches) if matches else None,
            "any_category_match_in_top3": any(category_matches),
            "top1_mapped_match": matches[0] if matches else None,
        })

    n = len(results) or 1
    mapped_results = [r for r in results if r["has_mapping"]]
    mn = len(mapped_results) or 1

    return {
        "kb_retrieval": {
            "top1_mapped_accuracy": round(
                sum(1 for r in mapped_results if r.get("top1_mapped_match")) / mn, 4
            ),
            "top3_mapped_hit_rate": round(
                sum(1 for r in mapped_results if r.get("any_mapped_match_in_top3")) / mn, 4
            ),
            "top3_category_hit_rate": round(
                sum(1 for r in results if r["any_category_match_in_top3"]) / n, 4
            ),
            "num_queries": len(results),
            "num_with_mapping": len(mapped_results),
            "per_query": results,
        }
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_sample_tickets()
    print(f"Loaded {len(samples)} sample tickets for evaluation.")

    print("\nEvaluating ticket retrieval...")
    ticket_metrics = evaluate_ticket_retrieval(samples)
    tr = ticket_metrics["ticket_retrieval"]
    print(f"  Precision@1: {tr['mean_precision@1']:.2%}")
    print(f"  Precision@3: {tr['mean_precision@3']:.2%}")
    print(f"  Precision@5: {tr['mean_precision@5']:.2%}")
    print(f"  MRR: {tr['mean_mrr']:.4f}")

    print("\nEvaluating KB retrieval (hybrid)...")
    kb_metrics = evaluate_kb_retrieval(samples)
    kb = kb_metrics["kb_retrieval"]
    print(f"  Top-1 mapped accuracy: {kb['top1_mapped_accuracy']:.2%}")
    print(f"  Top-3 mapped hit rate: {kb['top3_mapped_hit_rate']:.2%}")
    print(f"  Top-3 category hit rate: {kb['top3_category_hit_rate']:.2%}")

    combined = {**ticket_metrics, **kb_metrics}
    out_path = RESULTS_DIR / "retrieval_metrics.json"
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
