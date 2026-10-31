"""Ticket classification engine with LLM + keyword fallback + multi-category support."""

from __future__ import annotations

import json
import re
import time
from typing import Optional

from src.config import (
    CONFIDENCE_THRESHOLD,
    TAXONOMY,
    ClassificationResult,
    Ticket,
    get_llm_client,
    LLM_MODEL,
)

# Flatten taxonomy for validation
_VALID_SUBCATEGORIES: dict[str, str] = {}
for cat, subs in TAXONOMY.items():
    for sub in subs:
        _VALID_SUBCATEGORIES[sub] = cat

KEYWORD_RULES: list[tuple[list[str], str, str]] = [
    (["vpn", "tunnel", "remote access"], "Network", "VPN"),
    (["wireless", "wifi", "wi-fi", "ssid", "wlan"], "Network", "Wireless"),
    (["dns", "nslookup", "name resolution", "nxdomain"], "Network", "DNS"),
    (["dhcp", "lease", "ip assignment"], "Network", "DHCP"),
    (["ip address", "subnet", "ip conflict", "static ip"], "Network", "IP Address"),
    (["db2"], "Database", "DB2"),
    (["sql server", "mssql", "microsoft sql", "sqlcmd"], "Database", "MS SQL Server"),
    (["oracle", "ora-", "tnsnames", "sqlplus"], "Database", "Oracle"),
    (["cpu", "processor", "overheating", "fan noise", "thermal"], "Hardware", "CPU"),
    (["disk", "hard drive", "ssd", "storage", "hdd"], "Hardware", "Disk"),
    (["keyboard", "key stuck", "typing"], "Hardware", "Keyboard"),
    (["memory", "ram", "dimm", "out of memory"], "Hardware", "Memory"),
    (["monitor", "display", "screen", "hdmi", "displayport", "no signal"], "Hardware", "Monitor"),
    (["mouse", "cursor", "trackpad", "touchpad"], "Hardware", "Mouse"),
    (["email", "outlook", "mail", "smtp", "imap", "exchange"], "Software", "Email"),
    (["operating system", "windows", "os update", "boot", "bsod", "blue screen", "startup"], "Software", "Operating System"),
    (["antivirus", "virus", "malware", "trojan", "quarantine", "threat"], "Inquiry/Help", "Antivirus"),
    (["application", "app crash", "internal app", "software error"], "Inquiry/Help", "Internal Application"),
]

PRIORITY_KEYWORDS = {
    "P1": ["outage", "down for all", "security breach", "data loss", "critical system", "affecting all users", "complete failure"],
    "P2": ["degraded", "slow", "intermittent", "workaround", "multiple users", "major"],
    "P3": ["single user", "minor", "one person", "my laptop", "my computer"],
    "P4": ["cosmetic", "nice to have", "informational", "how to", "question", "inquiry"],
}


def _extract_signals(text: str) -> dict:
    text_lower = text.lower()
    error_codes = re.findall(r'0x[A-Fa-f0-9]{4,}|ERR-\d+|ORA-\d+|SQL\s*(?:Server\s*)?error\s*\d+|\d{3}\s+(?:error|forbidden|not found)', text, re.IGNORECASE)
    urgency = [w for w in ["urgent", "critical", "asap", "down", "outage", "emergency", "immediately"]
               if w in text_lower]
    user_match = re.search(r'(\d+)\s*users?\s*affected', text_lower)
    affected_users = int(user_match.group(1)) if user_match else None
    return {
        "error_codes": error_codes,
        "urgency_markers": urgency,
        "affected_users": affected_users,
    }


def _classify_priority_keyword(text: str) -> str:
    text_lower = text.lower()
    for priority, keywords in PRIORITY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return priority
    return "P3"


def classify_ticket_keyword_fallback(ticket: Ticket) -> ClassificationResult:
    """Keyword-based classification fallback — never crashes."""
    text = f"{ticket.title} {ticket.description}".lower()

    primary_cat, primary_sub = "Inquiry/Help", "Internal Application"
    secondary_cat, secondary_sub = None, None
    matches: list[tuple[str, str, int]] = []

    for keywords, cat, sub in KEYWORD_RULES:
        count = sum(1 for kw in keywords if kw in text)
        if count > 0:
            matches.append((cat, sub, count))

    matches.sort(key=lambda x: x[2], reverse=True)
    if matches:
        primary_cat, primary_sub = matches[0][0], matches[0][1]
        if len(matches) > 1 and matches[1][0] != primary_cat:
            secondary_cat, secondary_sub = matches[1][0], matches[1][1]

    priority = _classify_priority_keyword(f"{ticket.title} {ticket.description}")

    return ClassificationResult(
        category=primary_cat,
        subcategory=primary_sub,
        priority=priority,
        confidence=0.5,
        needs_review=True,
        reasoning="Classified using keyword fallback due to LLM unavailability.",
        secondary_category=secondary_cat,
        secondary_subcategory=secondary_sub,
        secondary_confidence=0.3 if secondary_cat else None,
    )


_SYSTEM_PROMPT = """You are an IT ticket classification engine. Classify the ticket into exactly one primary category and subcategory from this taxonomy:

Database: DB2, MS SQL Server, Oracle
Hardware: CPU, Disk, Keyboard, Memory, Monitor, Mouse
Network: DHCP, DNS, IP Address, VPN, Wireless
Software: Email, Operating System
Inquiry/Help: Antivirus, Internal Application

Return JSON with these fields:
{
  "category": "...",
  "subcategory": "...",
  "priority": "P1|P2|P3|P4",
  "confidence": 0.0-1.0,
  "reasoning": "1-2 sentences citing specific phrases from the ticket",
  "secondary_category": null or "...",
  "secondary_subcategory": null or "...",
  "secondary_confidence": null or 0.0-1.0
}

Priority: P1=complete outage/security breach/many users, P2=major degradation/workaround exists, P3=minor/single user, P4=cosmetic/informational.

If the ticket involves TWO distinct issues from different categories, provide secondary_category, secondary_subcategory, and secondary_confidence. Only if the second issue is clearly present (confidence > 0.4). If single issue, set these to null."""


def classify_ticket(ticket: Ticket) -> ClassificationResult:
    """Classify a ticket using LLM with keyword fallback."""
    signals = _extract_signals(f"{ticket.title} {ticket.description}")

    signal_text = ""
    if signals["error_codes"]:
        signal_text += f"\nDetected error codes: {', '.join(signals['error_codes'])}"
    if signals["urgency_markers"]:
        signal_text += f"\nUrgency markers: {', '.join(signals['urgency_markers'])}"
    if signals["affected_users"]:
        signal_text += f"\nAffected users: {signals['affected_users']}"

    user_msg = f"Title: {ticket.title}\nDescription: {ticket.description}"
    if signal_text:
        user_msg += f"\n\nExtracted signals:{signal_text}"

    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
            timeout=15,
        )
        raw = json.loads(response.choices[0].message.content)

        category = str(raw.get("category", "")).strip()
        subcategory = str(raw.get("subcategory", "")).strip()
        priority = str(raw.get("priority", "P3")).strip()
        confidence = float(raw.get("confidence", 0.5))
        reasoning = str(raw.get("reasoning", ""))

        if subcategory not in _VALID_SUBCATEGORIES or _VALID_SUBCATEGORIES.get(subcategory) != category:
            fallback = classify_ticket_keyword_fallback(ticket)
            fallback.reasoning = f"LLM returned invalid category ({category}/{subcategory}). " + fallback.reasoning
            return fallback

        if priority not in ("P1", "P2", "P3", "P4"):
            priority = "P3"
        confidence = max(0.0, min(1.0, confidence))

        sec_cat = raw.get("secondary_category")
        sec_sub = raw.get("secondary_subcategory")
        sec_conf = raw.get("secondary_confidence")
        if sec_cat and sec_sub:
            sec_cat = str(sec_cat).strip()
            sec_sub = str(sec_sub).strip()
            sec_conf = float(sec_conf) if sec_conf else 0.0
            if sec_sub not in _VALID_SUBCATEGORIES or _VALID_SUBCATEGORIES.get(sec_sub) != sec_cat or sec_conf <= 0.4:
                sec_cat, sec_sub, sec_conf = None, None, None
        else:
            sec_cat, sec_sub, sec_conf = None, None, None

        return ClassificationResult(
            category=category,
            subcategory=subcategory,
            priority=priority,
            confidence=confidence,
            needs_review=confidence < CONFIDENCE_THRESHOLD,
            reasoning=reasoning,
            secondary_category=sec_cat,
            secondary_subcategory=sec_sub,
            secondary_confidence=sec_conf,
        )

    except Exception:
        return classify_ticket_keyword_fallback(ticket)


def classify_batch(tickets: list[Ticket], delay: float = 0.5) -> list[ClassificationResult]:
    """Classify a batch of tickets with rate limiting."""
    results = []
    for i, ticket in enumerate(tickets):
        results.append(classify_ticket(ticket))
        if i < len(tickets) - 1:
            time.sleep(delay)
    return results
