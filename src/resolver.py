"""Resolution recommendation engine — synthesizes KB articles into actionable steps."""

from __future__ import annotations

import json
import re

import src.config as _cfg
from src.config import (
    ClassificationResult,
    Evidence,
    KnowledgeArticle,
    ResolutionRecommendation,
    SimilarTicket,
    Ticket,
    get_llm_client,
)

_ESCALATION_KEYWORDS = [
    "security breach", "data loss", "data leak", "compliance", "ransomware",
    "unauthorized access", "multiple users affected", "affecting all",
]

_SYSTEM_PROMPT = """You are an IT support resolution engine. Given a ticket, its classification, similar past tickets, and knowledge base articles, synthesize a step-by-step resolution.

RULES:
1. Base your resolution ONLY on the provided KB articles. Do not invent steps.
2. Cite which KB article each step comes from using [KB-XXXX].
3. Adapt the KB steps to the specific ticket details.
4. If the ticket spans two categories, address both — resolve the blocking issue first.
5. Suggest alternatives from other matching KB articles.
6. Determine if escalation is needed based on the KB articles' escalation criteria.

Return JSON:
{
  "recommended_resolution": "Step-by-step resolution text with [KB-XXXX] citations",
  "confidence": 0.0-1.0,
  "evidence": [
    {"source_id": "KB-XXXX", "source_type": "knowledge_base", "relevant_excerpt": "...", "support_reason": "..."},
    ...
  ],
  "alternative_resolutions": ["Alternative approach 1", ...],
  "needs_escalation": true/false,
  "knowledge_articles_used": ["KB-XXXX", ...]
}"""


def _build_resolution_prompt(
    ticket: Ticket,
    classification: ClassificationResult,
    similar_tickets: list[SimilarTicket],
    knowledge_articles: list[KnowledgeArticle],
) -> str:
    parts = [
        f"## Ticket\nTitle: {ticket.title}\nDescription: {ticket.description}",
        f"\n## Classification\nCategory: {classification.category}/{classification.subcategory}\n"
        f"Priority: {classification.priority} | Confidence: {classification.confidence:.2f}",
    ]

    if classification.secondary_category:
        parts.append(
            f"Secondary issue: {classification.secondary_category}/{classification.secondary_subcategory} "
            f"(confidence: {classification.secondary_confidence:.2f})\n"
            "NOTE: This ticket involves two distinct issues. Provide a combined resolution — "
            "resolve the blocking issue first."
        )

    if similar_tickets:
        st_lines = []
        for s in similar_tickets[:3]:
            st_lines.append(f"- [{s.ticket.id}] {s.ticket.category}/{s.ticket.subcategory} "
                          f"(similarity: {s.similarity_score:.0%}): {s.ticket.title[:80]}")
        parts.append(f"\n## Similar Past Tickets\n" + "\n".join(st_lines))

    if knowledge_articles:
        kb_lines = []
        for a in knowledge_articles:
            kb_lines.append(
                f"\n### {a.kb_id}: {a.title}\n"
                f"Category: {a.category}/{a.subcategory} | Relevance: {a.relevance_score:.2f}\n"
                f"Diagnostic Steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(a.diagnostic_steps)) +
                f"\nResolution Steps:\n" + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(a.resolution_steps)) +
                f"\nEscalation Criteria: {a.escalation_criteria}"
            )
        parts.append(f"\n## Knowledge Base Articles" + "".join(kb_lines))

    return "\n".join(parts)


def _needs_escalation_check(
    ticket: Ticket,
    classification: ClassificationResult,
    knowledge_articles: list[KnowledgeArticle],
) -> bool:
    if not knowledge_articles or all(a.relevance_score < 0.3 for a in knowledge_articles):
        return True
    if classification.confidence < 0.5:
        return True
    text_lower = f"{ticket.title} {ticket.description}".lower()
    if any(kw in text_lower for kw in _ESCALATION_KEYWORDS):
        return True
    return False


def _fallback_resolution(
    knowledge_articles: list[KnowledgeArticle],
) -> ResolutionRecommendation:
    """Return raw KB steps when LLM is unavailable."""
    if not knowledge_articles:
        return ResolutionRecommendation(
            recommended_resolution="No matching knowledge base articles found. This appears to be a novel issue. Recommended: escalate to L2/L3 support for manual investigation.",
            confidence=0.1,
            evidence=[],
            alternative_resolutions=[],
            needs_escalation=True,
            knowledge_articles_used=[],
        )

    top = knowledge_articles[0]
    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(top.resolution_steps))
    resolution = (
        f"Resolution sourced directly from {top.kb_id}: {top.title} "
        f"(LLM unavailable — showing raw KB steps):\n\n{steps}"
    )
    evidence = [Evidence(
        source_id=top.kb_id,
        source_type="knowledge_base",
        relevant_excerpt=top.resolution_steps[0] if top.resolution_steps else "",
        support_reason=f"Direct match from KB article: {top.title}",
    )]
    alternatives = []
    for a in knowledge_articles[1:3]:
        alt_steps = "; ".join(a.resolution_steps[:3])
        alternatives.append(f"{a.kb_id}: {a.title} — {alt_steps}")

    return ResolutionRecommendation(
        recommended_resolution=resolution,
        confidence=top.relevance_score * 0.6,
        evidence=evidence,
        alternative_resolutions=alternatives,
        needs_escalation=not knowledge_articles or all(a.relevance_score < 0.3 for a in knowledge_articles),
        knowledge_articles_used=[a.kb_id for a in knowledge_articles[:3]],
    )


def recommend_resolution(
    ticket: Ticket,
    classification: ClassificationResult,
    similar_tickets: list[SimilarTicket],
    knowledge_articles: list[KnowledgeArticle],
) -> ResolutionRecommendation:
    """Synthesize a resolution recommendation from KB articles and ticket context."""

    if not knowledge_articles or all(a.relevance_score < 0.3 for a in knowledge_articles):
        return ResolutionRecommendation(
            recommended_resolution="No matching knowledge base article found. This appears to be a novel issue type. Recommended: escalate to L2/L3 support for manual investigation.",
            confidence=0.1,
            evidence=[],
            alternative_resolutions=[],
            needs_escalation=True,
            knowledge_articles_used=[],
        )

    prompt = _build_resolution_prompt(ticket, classification, similar_tickets, knowledge_articles)

    try:
        client = get_llm_client()
        try:
            response = client.chat.completions.create(
                model=_cfg.LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                timeout=20,
            )
        except Exception as model_err:
            if "404" in str(model_err) or "NotFoundError" in type(model_err).__name__:
                # Model not accessible; auto-retry with standard 8B instant model
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    timeout=20,
                )
            else:
                raise model_err

        raw = json.loads(response.choices[0].message.content)

        evidence = []
        for e in raw.get("evidence", []):
            evidence.append(Evidence(
                source_id=str(e.get("source_id", "")),
                source_type=str(e.get("source_type", "knowledge_base")),
                relevant_excerpt=str(e.get("relevant_excerpt", "")),
                support_reason=str(e.get("support_reason", "")),
            ))

        needs_escalation = raw.get("needs_escalation", False) or _needs_escalation_check(
            ticket, classification, knowledge_articles
        )

        return ResolutionRecommendation(
            recommended_resolution=str(raw.get("recommended_resolution", "")),
            confidence=max(0.0, min(1.0, float(raw.get("confidence", 0.5)))),
            evidence=evidence,
            alternative_resolutions=[str(a) for a in raw.get("alternative_resolutions", [])],
            needs_escalation=needs_escalation,
            knowledge_articles_used=[str(k) for k in raw.get("knowledge_articles_used", [])],
        )

    except Exception as e:
        rec = _fallback_resolution(knowledge_articles)
        rec.recommended_resolution = f"[Note: LLM synthesis error ({type(e).__name__}: {e}). Showing raw KB steps below]\n\n" + rec.recommended_resolution
        return rec
