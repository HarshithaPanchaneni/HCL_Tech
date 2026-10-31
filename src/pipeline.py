import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from src.config import (
    Ticket,
    AnalysisResult,
    ResolutionRecommendation,
    Evidence,
    FeedbackRecord,
)
from src.classifier_adapter import classify_for_pipeline
from src.retriever import find_similar_tickets, find_knowledge_articles
from src.llm import generate_resolution


FEEDBACK_PATH = Path("data/processed/feedback.json")


def analyze_ticket(ticket: Ticket) -> AnalysisResult:
    """Run classification, retrieval and resolution for a ticket."""

    timings = {}
    total_start = __import__("time").time()

    if not ticket.id:
        ticket.id = str(uuid4())

    ticket_text = f"{ticket.title}\n{ticket.description}"

    # 1. Classify the ticket.
    start = __import__("time").time()
    classification = classify_for_pipeline(ticket_text)
    timings["classification_ms"] = round((__import__("time").time() - start) * 1000)

    ticket.category = classification.category
    ticket.subcategory = classification.subcategory
    ticket.priority = classification.priority

    # 2. Retrieve similar historical tickets.
    start = __import__("time").time()
    similar_tickets = find_similar_tickets(
        ticket, top_k=5,
        category_filter=classification.category
    )
    timings["ticket_retrieval_ms"] = round((__import__("time").time() - start) * 1000)

    # 3. Retrieve relevant knowledge-base articles.
    start = __import__("time").time()
    knowledge_articles = find_knowledge_articles(
        ticket, classification=classification, top_k=3
    )
    timings["kb_retrieval_ms"] = round((__import__("time").time() - start) * 1000)

    # 4. Generate a resolution using retrieved KB articles.
    start = __import__("time").time()

    if knowledge_articles:
        recommended_resolution = generate_resolution(ticket_text, knowledge_articles)
    else:
        recommended_resolution = (
            "No sufficiently relevant knowledge-base article was found. "
            "Please escalate this ticket to the appropriate support team."
        )

    evidence = [
        Evidence(
            source_id=article.kb_id,
            source_type="knowledge_base",
            relevant_excerpt=(
                article.resolution_steps[0] if article.resolution_steps else ""
            ),
            support_reason=f"Relevant KB article: {article.title}",
        )
        for article in knowledge_articles
    ]

    resolution = ResolutionRecommendation(
        recommended_resolution=recommended_resolution,
        confidence=classification.confidence,
        evidence=evidence,
        alternative_resolutions=[],
        needs_escalation=(
            classification.needs_review or not knowledge_articles
        ),
        knowledge_articles_used=[a.kb_id for a in knowledge_articles],
    )

    timings["resolution_ms"] = round((__import__("time").time() - start) * 1000)
    timings["total_ms"] = round((__import__("time").time() - total_start) * 1000)
    ticket.metadata["timings"] = timings

    return AnalysisResult(
        input_ticket=ticket,
        classification=classification,
        similar_tickets=similar_tickets,
        knowledge_articles=knowledge_articles,
        resolution=resolution,
    )


def record_feedback(feedback: FeedbackRecord) -> None:
    """Append analyst feedback to the feedback log."""

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        records = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        records = []

    records.append(asdict(feedback))
    FEEDBACK_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )