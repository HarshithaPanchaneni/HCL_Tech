"""Orchestration pipeline: classify → retrieve → resolve.

Usage:
    from src.pipeline import analyze_ticket
    result = analyze_ticket(ticket)

Set MOCK_MODE=true to use hardcoded results for UI development.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from src.config import (
    AnalysisResult,
    ClassificationResult,
    Evidence,
    FeedbackRecord,
    KnowledgeArticle,
    ResolutionRecommendation,
    SimilarTicket,
    Ticket,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = PROJECT_ROOT / "data" / "processed" / "feedback.json"

MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"


def _mock_analyze_ticket(ticket: Ticket) -> AnalysisResult:
    classification = ClassificationResult(
        category="Network",
        subcategory="VPN",
        priority="P2",
        confidence=0.92,
        needs_review=False,
        reasoning="Ticket mentions 'VPN' and 'Gateway not responding', clearly indicating a VPN connectivity issue.",
        secondary_category=None,
        secondary_subcategory=None,
        secondary_confidence=None,
    )
    similar_tickets = [
        SimilarTicket(
            ticket=Ticket(id="TKT-1351", title="VPN keeps dropping randomly", description="My VPN keeps dropping randomly, making it impossible to work remotely.", category="Network", subcategory="VPN"),
            similarity_score=0.87,
            matching_reason="Network/VPN, 87% similarity",
        ),
        SimilarTicket(
            ticket=Ticket(id="TKT-1307", title="VPN connection times out", description="VPN si blocca costantemente, costringendomi a riavviare.", category="Network", subcategory="VPN"),
            similarity_score=0.82,
            matching_reason="Network/VPN, 82% similarity",
        ),
        SimilarTicket(
            ticket=Ticket(id="TKT-1357", title="VPN crashes repeatedly", description="My VPN keeps crashing, forcing me to restart my device.", category="Network", subcategory="VPN"),
            similarity_score=0.79,
            matching_reason="Network/VPN, 79% similarity",
        ),
    ]
    knowledge_articles = [
        KnowledgeArticle(
            kb_id="KB-0018",
            title="VPN will not connect or repeatedly disconnects",
            category="Network",
            subcategory="VPN",
            symptoms=["Cannot connect to company VPN from home", "VPN connects but disconnects every few minutes", "Sign-in or gateway error with working internet"],
            resolution_steps=[
                "Restart the VPN client and reconnect using the organization's published gateway/profile.",
                "Confirm automatic date/time is enabled and complete the normal corporate sign-in/MFA flow.",
                "Install the approved VPN client update from the company portal if out of date.",
                "If certificate or compliance issue, use official device-enrollment workflow.",
                "Capture client diagnostic bundle and timestamp if still failing.",
            ],
            diagnostic_steps=[
                "Confirm ordinary internet access by opening an approved public site.",
                "Record VPN client name/version, gateway, exact error, and connection time.",
                "Check system date/time and corporate authentication.",
                "Review VPN client logs and service-health notices.",
            ],
            escalation_criteria="Escalate if VPN gateway is unavailable, many users affected, or authentication errors persist.",
            relevance_score=0.92,
            retrieval_source="subcategory_map",
        ),
        KnowledgeArticle(
            kb_id="KB-0019",
            title="VPN is slow because of routing, gateway load, or tunnel configuration",
            category="Network",
            subcategory="VPN",
            symptoms=["Web pages much slower while VPN connected", "Only some internal apps are slow", "Video calls unreliable when connected remotely"],
            resolution_steps=[
                "Disconnect bandwidth-heavy nonessential activity.",
                "Restart VPN client and confirm latest approved version.",
                "Test affected application and a second internal resource.",
                "Provide logs to network operations for route/gateway analysis.",
            ],
            diagnostic_steps=[
                "Compare latency with and without VPN.",
                "Check if all traffic or specific app is slow.",
                "Run ping to internal gateway.",
            ],
            escalation_criteria="Escalate if performance prevents critical work or many remote users affected.",
            relevance_score=0.71,
            retrieval_source="subcategory_map",
        ),
    ]
    resolution = ResolutionRecommendation(
        recommended_resolution=(
            "Based on KB-0018 (VPN connection failures), follow these steps:\n\n"
            "1. **Restart VPN client** — Close the VPN application completely and relaunch it. "
            "Reconnect using the organization's published gateway profile. [KB-0018]\n\n"
            "2. **Verify system time** — Ensure automatic date/time is enabled in system settings. "
            "Incorrect time can cause certificate validation failures. [KB-0018]\n\n"
            "3. **Check for client updates** — Open the company software portal and install any "
            "pending VPN client updates. Outdated clients may not support current gateway configurations. [KB-0018]\n\n"
            "4. **Test on alternate network** — If available, try connecting via an approved mobile "
            "hotspot to rule out local network issues. [KB-0018]\n\n"
            "5. **Capture diagnostics** — If the issue persists, export the VPN client diagnostic "
            "bundle with a timestamp and forward to network support. [KB-0018]"
        ),
        confidence=0.88,
        evidence=[
            Evidence(source_id="KB-0018", source_type="knowledge_base",
                     relevant_excerpt="Restart the VPN client and reconnect using the organization's published gateway/profile.",
                     support_reason="Direct resolution step for VPN connection failures."),
            Evidence(source_id="KB-0018", source_type="knowledge_base",
                     relevant_excerpt="Confirm automatic date/time is enabled and complete the normal corporate sign-in/MFA flow.",
                     support_reason="Gateway errors can be caused by time synchronization issues affecting certificate validation."),
            Evidence(source_id="TKT-1351", source_type="ticket",
                     relevant_excerpt="VPN keeps dropping randomly",
                     support_reason="Similar pattern of intermittent VPN disconnections reported by other users."),
        ],
        alternative_resolutions=[
            "If VPN performance is the primary issue rather than connectivity, check KB-0019 for routing and gateway load troubleshooting.",
            "If the issue is specific to a certificate or device compliance problem, use the official device-enrollment or certificate-renewal workflow.",
        ],
        needs_escalation=False,
        knowledge_articles_used=["KB-0018", "KB-0019"],
    )
    return AnalysisResult(
        input_ticket=ticket,
        classification=classification,
        similar_tickets=similar_tickets,
        knowledge_articles=knowledge_articles,
        resolution=resolution,
    )


def analyze_ticket(ticket: Ticket) -> AnalysisResult:
    """Main entry point — orchestrates classification, retrieval, and resolution."""
    if MOCK_MODE:
        return _mock_analyze_ticket(ticket)

    timings: dict[str, float] = {}
    start_total = time.time()

    # Step 1: Classification
    try:
        t0 = time.time()
        from src.classifier import classify_ticket
        classification = classify_ticket(ticket)
        timings["classification_ms"] = round((time.time() - t0) * 1000)
    except Exception as e:
        logger.warning("Classification failed: %s — using keyword fallback", e)
        try:
            from src.classifier import classify_ticket_keyword_fallback
            classification = classify_ticket_keyword_fallback(ticket)
        except Exception:
            classification = ClassificationResult(
                category="Unknown", subcategory="Unknown", priority="P3",
                confidence=0.0, needs_review=True,
                reasoning="Classification unavailable — all methods failed.",
            )
        timings["classification_ms"] = round((time.time() - start_total) * 1000)

    # Step 2: Ticket similarity retrieval
    similar_tickets: list[SimilarTicket] = []
    try:
        t0 = time.time()
        from src.retriever import find_similar_tickets
        category_filter = classification.category if classification.confidence > 0.8 else None
        similar_tickets = find_similar_tickets(ticket, top_k=5, category_filter=category_filter)
        timings["ticket_retrieval_ms"] = round((time.time() - t0) * 1000)
    except Exception as e:
        logger.warning("Ticket retrieval failed: %s", e)
        timings["ticket_retrieval_ms"] = 0

    # Step 3: KB article retrieval
    knowledge_articles: list[KnowledgeArticle] = []
    try:
        t0 = time.time()
        from src.retriever import find_knowledge_articles
        knowledge_articles = find_knowledge_articles(ticket, classification, top_k=3)
        timings["kb_retrieval_ms"] = round((time.time() - t0) * 1000)
        kb_modes = set(a.retrieval_source for a in knowledge_articles)
        timings["kb_retrieval_mode"] = "+".join(sorted(kb_modes)) if kb_modes else "none"
    except Exception as e:
        logger.warning("KB retrieval failed: %s", e)
        timings["kb_retrieval_ms"] = 0
        timings["kb_retrieval_mode"] = "failed"

    # Step 4: Resolution synthesis
    try:
        t0 = time.time()
        from src.resolver import recommend_resolution
        resolution = recommend_resolution(ticket, classification, similar_tickets, knowledge_articles)
        timings["resolution_ms"] = round((time.time() - t0) * 1000)
    except Exception as e:
        logger.warning("Resolution synthesis failed: %s", e)
        # Fallback: return raw KB steps
        if knowledge_articles:
            top = knowledge_articles[0]
            steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(top.resolution_steps))
            resolution = ResolutionRecommendation(
                recommended_resolution=f"Resolution from {top.kb_id}: {top.title} (raw KB steps — LLM unavailable):\n\n{steps}",
                confidence=top.relevance_score * 0.5,
                evidence=[Evidence(source_id=top.kb_id, source_type="knowledge_base",
                                   relevant_excerpt=top.resolution_steps[0] if top.resolution_steps else "",
                                   support_reason=f"Direct match from {top.title}")],
                alternative_resolutions=[],
                needs_escalation=False,
                knowledge_articles_used=[top.kb_id],
            )
        else:
            resolution = ResolutionRecommendation(
                recommended_resolution="System degraded — unable to generate resolution. Please escalate to IT support manually.",
                confidence=0.0,
                evidence=[],
                alternative_resolutions=[],
                needs_escalation=True,
                knowledge_articles_used=[],
            )
        timings["resolution_ms"] = round((time.time() - t0) * 1000)

    timings["total_ms"] = round((time.time() - start_total) * 1000)

    result = AnalysisResult(
        input_ticket=ticket,
        classification=classification,
        similar_tickets=similar_tickets,
        knowledge_articles=knowledge_articles,
        resolution=resolution,
    )
    # Attach timing metadata
    ticket.metadata["timings"] = timings
    return result


def record_feedback(feedback: FeedbackRecord) -> None:
    """Append feedback to the feedback log."""
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = []
    if FEEDBACK_PATH.is_file():
        try:
            records = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            records = []

    from dataclasses import asdict
    records.append(asdict(feedback))
    FEEDBACK_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
