"""Streamlit frontend for the Intelligent IT Ticket Analysis and Resolution Assistant."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    AnalysisResult,
    FeedbackRecord,
    Ticket,
)

st.set_page_config(
    page_title="IT Ticket Assistant",
    page_icon="🎫",
    layout="wide",
)

# ── Custom CSS ──

st.markdown("""
<style>
    .main-header { color: #0073e6; font-size: 2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { color: #666; font-size: 1.1rem; margin-top: 0; }
    .card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1rem;
        border-left: 4px solid #0073e6;
    }
    .card-escalation { border-left-color: #dc3545; background: #fff5f5; }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.85rem; font-weight: 600; margin-right: 6px;
    }
    .badge-blue { background: #e3f0ff; color: #0073e6; }
    .badge-green { background: #e6f9e6; color: #1a7f1a; }
    .badge-yellow { background: #fff8e1; color: #b38600; }
    .badge-red { background: #ffe6e6; color: #cc0000; }
    .badge-grey { background: #f0f0f0; color: #666; }
    .badge-orange { background: #fff0e0; color: #cc6600; }
    .evidence-box {
        background: #f8f9fa; border-left: 3px solid #0073e6;
        padding: 0.8rem; margin: 0.5rem 0; border-radius: 4px;
    }
    .evidence-box-ticket {
        border-left-color: #888;
    }
    .status-ok { color: #1a7f1a; }
    .status-warn { color: #b38600; }
    .status-fail { color: #cc0000; }
    .metric-bar {
        height: 24px; border-radius: 4px; display: inline-block;
        text-align: center; color: white; font-size: 0.75rem;
        line-height: 24px; min-width: 40px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load sample tickets ──

@st.cache_data
def load_sample_tickets() -> list[dict]:
    path = PROJECT_ROOT / "data" / "sample_tickets.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def run_analysis(ticket: Ticket) -> AnalysisResult:
    from src.pipeline import analyze_ticket
    return analyze_ticket(ticket)


# ── Sidebar ──

with st.sidebar:
    st.markdown("### 🎫 IT Ticket Assistant")
    st.markdown("---")

    samples = load_sample_tickets()
    sample_names = ["(Enter manually)"] + [f"{s['id']}: {s['title'][:40]}" for s in samples]
    selected = st.selectbox("📋 Sample Tickets", sample_names)

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "**Two-Source RAG** system that separates ticket similarity search "
        "from knowledge-based resolution. Evidence-backed, auditable, no hallucinated fixes."
    )
    st.markdown("---")
    st.markdown("### Tech Stack")
    st.markdown(
        "- **LLM**: Groq (Llama 3.1 70B)\n"
        "- **Embeddings**: Multilingual MiniLM\n"
        "- **Vector DB**: ChromaDB\n"
        "- **UI**: Streamlit"
    )


# ── Header ──

st.markdown('<p class="main-header">🎫 Intelligent IT Ticket Analysis and Resolution Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Two-Source RAG — Evidence-backed resolutions from curated knowledge base</p>', unsafe_allow_html=True)
st.markdown("---")


# ── Section 1: Input ──

sample_idx = sample_names.index(selected) - 1 if selected != "(Enter manually)" else -1

if sample_idx >= 0:
    default_title = samples[sample_idx]["title"]
    default_desc = samples[sample_idx]["description"]
else:
    default_title = ""
    default_desc = ""

col_input1, col_input2 = st.columns([1, 2])
with col_input1:
    title = st.text_input("Ticket Title", value=default_title, placeholder="e.g., Can't connect to VPN")
with col_input2:
    description = st.text_area("Ticket Description", value=default_desc, height=120,
                               placeholder="Describe the issue in detail...")

analyze_btn = st.button("🔍 Analyze Ticket", type="primary", use_container_width=True)


# ── Analysis ──

if analyze_btn and (title.strip() or description.strip()):
    ticket = Ticket(
        id=f"NEW-{int(time.time()) % 100000}",
        title=title.strip(),
        description=description.strip(),
    )

    with st.spinner("Analyzing ticket..."):
        start = time.time()
        result = run_analysis(ticket)
        elapsed = time.time() - start

    st.session_state["result"] = result
    st.session_state["elapsed"] = elapsed
    st.session_state["ticket"] = ticket

elif analyze_btn:
    st.warning("Please enter a title or description.")


# ── Display results ──

if "result" in st.session_state:
    result: AnalysisResult = st.session_state["result"]
    elapsed: float = st.session_state["elapsed"]
    c = result.classification
    r = result.resolution

    st.markdown("---")

    # ── Section 2: Classification ──
    st.markdown("### 🏷️ Classification")

    priority_colors = {"P1": "red", "P2": "orange", "P3": "yellow", "P4": "green"}
    priority_icons = {"P1": "🔴", "P2": "🟠", "P3": "🟡", "P4": "🟢"}
    p_color = priority_colors.get(c.priority, "grey")
    p_icon = priority_icons.get(c.priority, "⚪")

    col_c1, col_c2, col_c3 = st.columns([2, 2, 1])
    with col_c1:
        st.markdown(f'<span class="badge badge-blue">{c.category}</span>'
                    f'<span class="badge badge-blue">{c.subcategory}</span>'
                    f'<span class="badge badge-{p_color}">{p_icon} {c.priority}</span>',
                    unsafe_allow_html=True)
    with col_c2:
        conf_color = "green" if c.confidence > 0.8 else ("orange" if c.confidence > 0.5 else "red")
        st.progress(c.confidence, text=f"Confidence: {c.confidence:.0%}")
    with col_c3:
        if c.needs_review:
            st.warning("⚠️ Needs Review")

    if c.reasoning:
        st.markdown(f"*{c.reasoning}*")

    if c.secondary_category and c.secondary_confidence and c.secondary_confidence > 0.4:
        st.markdown(f"**Secondary Issue Detected:** "
                    f'<span class="badge badge-yellow">{c.secondary_category}</span>'
                    f'<span class="badge badge-yellow">{c.secondary_subcategory}</span>'
                    f' (confidence: {c.secondary_confidence:.0%})',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 3: KB Articles ──
    st.markdown("### 📚 Knowledge Base Matches")

    if result.knowledge_articles:
        for a in result.knowledge_articles:
            score_color = "green" if a.relevance_score > 0.7 else ("yellow" if a.relevance_score > 0.4 else "grey")
            source_label = "⚡ Fast Path" if a.retrieval_source == "subcategory_map" else "🔍 Vector"
            with st.expander(f"**{a.kb_id}** — {a.title}  |  Relevance: {a.relevance_score:.0%}  |  {source_label}", expanded=False):
                st.markdown(f"**Category:** {a.category} / {a.subcategory}")
                st.markdown("**Symptoms:**")
                for s in a.symptoms:
                    st.markdown(f"- {s}")
                st.markdown("**Diagnostic Steps:**")
                for i, s in enumerate(a.diagnostic_steps, 1):
                    st.markdown(f"{i}. {s}")
                st.markdown("**Resolution Steps:**")
                for i, s in enumerate(a.resolution_steps, 1):
                    st.markdown(f"{i}. {s}")
                st.markdown(f"**Escalation Criteria:** {a.escalation_criteria}")
    else:
        st.info("⚠️ No matching KB articles found — this may be a novel issue type.")

    st.markdown("---")

    # ── Section 4: Similar Tickets ──
    st.markdown("### 🔗 Similar Tickets")

    if result.similar_tickets:
        for s in result.similar_tickets:
            score_pct = s.similarity_score * 100
            color = "green" if score_pct > 80 else ("yellow" if score_pct > 50 else "grey")
            st.markdown(
                f'<span class="badge badge-{color}">{score_pct:.0f}%</span> '
                f'**{s.ticket.id}** — {s.ticket.title[:80]} '
                f'<span class="badge badge-grey">{s.ticket.category}/{s.ticket.subcategory}</span>',
                unsafe_allow_html=True
            )
    else:
        st.info("No similar tickets found.")

    st.markdown("---")

    # ── Section 5: Resolution ──
    st.markdown("### 📋 Recommended Resolution")

    if r.needs_escalation:
        st.error("🚨 **Escalation Recommended** — This issue requires L2/L3 support attention.")

    # Resolution source indicator
    if r.evidence:
        st.success("🟢 **LLM-Synthesized Resolution** — Evidence-backed from knowledge base")
    elif r.knowledge_articles_used:
        st.warning("🟡 **Raw KB Match** — LLM unavailable, showing KB article steps directly")
    else:
        st.error("🔴 **No KB Match** — Novel issue, escalation required")

    st.markdown(r.recommended_resolution)

    st.markdown(f"**Resolution Confidence:** {r.confidence:.0%}")

    # Evidence trail
    if r.evidence:
        st.markdown("#### 🔍 Evidence Trail")
        st.caption("*Traceability: every recommendation links back to its source*")
        for e in r.evidence:
            box_class = "evidence-box" if e.source_type == "knowledge_base" else "evidence-box evidence-box-ticket"
            badge_class = "badge-blue" if e.source_type == "knowledge_base" else "badge-grey"
            st.markdown(
                f'<div class="{box_class}">'
                f'<span class="badge {badge_class}">{e.source_id}</span> '
                f'<em>{e.source_type.replace("_", " ").title()}</em><br>'
                f'<blockquote>{e.relevant_excerpt}</blockquote>'
                f'<small>{e.support_reason}</small>'
                f'</div>',
                unsafe_allow_html=True
            )
    elif r.needs_escalation:
        st.info("⚠️ No evidence sources matched. This is a novel issue not covered by the knowledge base.")

    if r.alternative_resolutions:
        with st.expander("Alternative Resolutions"):
            for alt in r.alternative_resolutions:
                st.markdown(f"- {alt}")

    if r.knowledge_articles_used:
        st.markdown(f"**KB Articles Used:** {', '.join(r.knowledge_articles_used)}")

    # ── Section 5b: System Status ──
    timings = result.input_ticket.metadata.get("timings", {})

    status_items = []
    status_items.append(f'<span class="status-ok">✅ Classification</span>')
    status_items.append(f'<span class="{"status-ok" if result.similar_tickets else "status-warn"}">{"✅" if result.similar_tickets else "⚠️"} Ticket Retrieval</span>')
    kb_status = "status-ok" if result.knowledge_articles else "status-warn"
    status_items.append(f'<span class="{kb_status}">{"✅" if result.knowledge_articles else "⚠️"} KB Retrieval</span>')
    llm_ok = bool(r.evidence)
    status_items.append(f'<span class="{"status-ok" if llm_ok else "status-warn"}">{"✅" if llm_ok else "⚠️"} LLM Synthesis</span>')

    st.markdown(" | ".join(status_items), unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 6: Feedback ──
    st.markdown("### 💬 Analyst Feedback")

    col_f1, col_f2, col_f3 = st.columns(3)
    ticket_id = result.input_ticket.id

    with col_f1:
        if st.button("✅ Useful", key="fb_useful"):
            from src.pipeline import record_feedback
            record_feedback(FeedbackRecord(ticket_id=ticket_id, feedback_type="useful", resolution_helpful=True))
            st.toast("Feedback recorded: Useful ✅")

    with col_f2:
        if st.button("❌ Incorrect", key="fb_incorrect"):
            from src.pipeline import record_feedback
            record_feedback(FeedbackRecord(ticket_id=ticket_id, feedback_type="incorrect", resolution_helpful=False))
            st.toast("Feedback recorded: Incorrect ❌")

    with col_f3:
        if st.button("⏫ Needs Escalation", key="fb_escalate"):
            from src.pipeline import record_feedback
            record_feedback(FeedbackRecord(ticket_id=ticket_id, feedback_type="needs_escalation"))
            st.toast("Feedback recorded: Escalation ⏫")

    with st.expander("📝 Detailed Feedback"):
        fb_class_correct = st.radio("Was the classification correct?", ["Yes", "No"], horizontal=True, key="fb_class")
        fb_retrieval = st.radio("Were similar tickets relevant?", ["Yes", "No"], horizontal=True, key="fb_ret")
        fb_resolution = st.radio("Was the resolution helpful?", ["Yes", "No"], horizontal=True, key="fb_res")
        fb_missing_topic = st.text_input("Missing topic in knowledge base?", key="fb_missing")
        fb_notes = st.text_area("Additional notes", key="fb_notes", height=80)

        if st.button("Submit Detailed Feedback", key="fb_submit"):
            from src.pipeline import record_feedback
            record_feedback(FeedbackRecord(
                ticket_id=ticket_id,
                feedback_type="detailed",
                classification_correct=fb_class_correct == "Yes",
                retrieval_relevant=fb_retrieval == "Yes",
                resolution_helpful=fb_resolution == "Yes",
                missing_kb_topic=fb_missing_topic or None,
                analyst_notes=fb_notes or None,
            ))
            st.toast("Detailed feedback submitted! 📝")

    st.markdown("---")

    # ── Section 7: Performance & Audit ──
    with st.expander("⏱️ Performance & Audit Log"):
        if timings:
            st.markdown("**Processing Time Breakdown:**")
            cols = st.columns(5)
            labels = [
                ("Classification", timings.get("classification_ms", 0)),
                ("Ticket Retrieval", timings.get("ticket_retrieval_ms", 0)),
                ("KB Retrieval", timings.get("kb_retrieval_ms", 0)),
                ("Resolution", timings.get("resolution_ms", 0)),
                ("Total", timings.get("total_ms", 0)),
            ]
            for col, (label, ms) in zip(cols, labels):
                col.metric(label, f"{ms}ms")

            kb_mode = timings.get("kb_retrieval_mode", "unknown")
            if "subcategory_map" in kb_mode:
                st.markdown("⚡ **KB Retrieval: Fast Path** (subcategory mapping)")
            elif "vector" in kb_mode:
                st.markdown("🔍 **KB Retrieval: Vector Search** (semantic similarity)")
            else:
                st.markdown(f"KB Retrieval Mode: {kb_mode}")
        else:
            st.markdown(f"**Total time:** {elapsed:.2f}s")

        st.markdown("**Full Analysis Result (JSON):**")
        st.json(asdict(result))

        st.download_button(
            "📥 Export as JSON",
            data=json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str),
            file_name=f"analysis_{result.input_ticket.id}.json",
            mime="application/json",
        )
