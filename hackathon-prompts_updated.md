# HCL Tech Hackathon — AI Agent Prompts for 4-Person Team

**Repo:** https://github.com/HarshithaPanchaneni/HCL_Tech.git
**Problem:** Use Case 1 — Intelligent IT Ticket Analysis and Resolution Assistant

**Architecture:** Two-Source RAG — Ticket corpus (similarity/classification) + Knowledge Base (resolution)
**Why two sources:** The HuggingFace dataset has NO resolution text. Fabricating resolutions would undermine traceability. Instead, a curated Knowledge Base of 30 runbook articles provides evidence-backed resolutions. This mirrors HCL's own production system (RAG over KEDB + runbooks + tickets on Azure with OpenAI, handling 25K+ monthly tickets).

**CRITICAL — Dataset Reality (discovered after inspection):**
- The actual 18 subcategories are DIFFERENT from the HuggingFace page description:
  - Database: **DB2, MS SQL Server, Oracle** (not "DB performance/access/backup")
  - Hardware: **CPU, Disk, Keyboard, Memory, Monitor, Mouse** (not "Laptop/Printer/Docking station")
  - Network: **DHCP, DNS, IP Address, VPN, Wireless** (not "WiFi/Firewall/Network drive")
  - Software: **Email, Operating System** (not "Installation issue/Application error")
  - Inquiry/Help: **Antivirus, Internal Application** (not "General inquiry/How-to question")
- Columns: `Category`, `Subcategory`, `Text` (single text field, no separate title)
- 100 tickets per subcategory, 1800 total

**LLM Setup (no OpenAI needed):**
- **Primary:** Groq API (free, fast) — `llama-3.1-70b-versatile` via OpenAI-compatible endpoint
- **Local dev:** Ollama — `qwen2.5:7b` on localhost, same `openai` Python package
- Both use the same code, just different `base_url` — swappable via `LLM_PROVIDER` env var

**3-Layer Fallback Strategy:**
1. **LLM fails (Groq timeout/rate limit):** Classifier falls back to keyword-based matching (confidence capped at 0.5, `needs_review=True`). Resolver returns raw KB article steps without LLM synthesis.
2. **Groq entirely unavailable:** Switch to Ollama locally (`export LLM_PROVIDER=ollama`). Same code, different endpoint.
3. **Both LLMs unavailable:** Vector search is LOCAL (no LLM needed). System still returns similar tickets + matched KB articles. UI shows "⚠️ LLM unavailable — showing raw matches for manual review."

**Knowledge Gap Handling (ticket doesn't match any KB article):**
- If all KB articles score < 0.3 relevance → `needs_escalation = True`, response: "No matching knowledge base article. Novel issue — escalate to L2/L3."
- The LLM can still provide directional guidance from classification + similar ticket patterns.
- Feedback form has "Missing topic in knowledge base?" field → logs the gap for KB enrichment (continuous improvement loop, same as HCL's KEDB enrichment process).
- **Key selling point:** "Our system doesn't hallucinate resolutions. When it doesn't know, it says so and escalates."

**Multi-Category Ticket Handling (ticket spans 2 categories):**
- Example: "Can't connect to VPN and my laptop keeps crashing" → Network/VPN + Hardware/Laptop
- Classifier returns PRIMARY + SECONDARY classification with separate confidence scores
- Secondary is used when its confidence > 0.4 (not a noise match)
- Retrieval fans out: KB search + ticket search run for BOTH categories, results merged and deduplicated
- Resolver synthesizes a combined resolution: "First resolve VPN connectivity (KB-0018), then address laptop instability (KB-0006)"
- UI shows both classification badges with individual confidence scores

Everyone: Clone the repo first, create your branch, work there, then push. We merge into `main` at integration checkpoints.

```bash
git clone https://github.com/HarshithaPanchaneni/HCL_Tech.git
cd HCL_Tech
git checkout -b <your-branch-name>
```

## Agreed Directory Structure

```
HCL_Tech/
├── README.md                    # Person 4 owns this
├── requirements.txt             # Shared — each person adds their deps
├── data/
│   ├── raw/                     # Person 1: downloaded datasets
│   ├── processed/               # Person 1: cleaned + embedded data (ChromaDB)
│   ├── knowledge_base/          # Person 1: 30 curated KB articles (articles.json)
│   └── sample_tickets.json      # Person 1: 5+ worked example tickets for demo
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py         # Person 1: data loading, embedding, vector store
│   ├── kb_indexer.py            # Person 1: knowledge base → ChromaDB indexing
│   ├── classifier.py            # Person 2: ticket classification engine (LLM)
│   ├── retriever.py             # Person 1: TWO retrieval functions (tickets + KB)
│   ├── resolver.py              # Person 3: resolution synthesis via RAG
│   ├── pipeline.py              # Person 3: orchestrates classify → retrieve → resolve
│   └── config.py                # Shared constants + dataclasses + LLM provider config
├── app/
│   ├── app.py                   # Person 4: Streamlit app
│   └── components/              # Person 4: UI components
├── eval/
│   ├── classification_eval.py   # Person 2: classification metrics
│   ├── retrieval_eval.py        # Person 1: retrieval metrics
│   └── results/                 # Eval outputs (confusion matrix, etc.)
├── docs/
│   ├── architecture.md          # Person 3
│   ├── eval_report.md           # Person 2 + Person 1
│   └── worked_examples.md       # Person 3
└── presentation/
    └── slides.md                # Person 4
```

## Shared Interface Contract

All components communicate via these Python dataclasses. **Every person must use these exact types.** Put this in `src/config.py`:

```python
import os
from dataclasses import dataclass, field
from typing import Optional

# === LLM PROVIDER CONFIG — Switch via environment variable ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "ollama"

if LLM_PROVIDER == "ollama":
    LLM_MODEL = "qwen2.5:7b"
    LLM_BASE_URL = "http://localhost:11434/v1"
    LLM_API_KEY = "ollama"  # Ollama ignores keys but openai package requires one
else:  # groq (default)
    LLM_MODEL = "llama-3.1-70b-versatile"
    LLM_BASE_URL = "https://api.groq.com/openai/v1"
    LLM_API_KEY = os.getenv("GROQ_API_KEY")

# === SHARED DATA MODELS — DO NOT CHANGE WITHOUT TELLING THE TEAM ===

@dataclass
class Ticket:
    id: str
    title: str
    description: str
    category: Optional[str] = None
    subcategory: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    resolution: Optional[str] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class ClassificationResult:
    category: str
    subcategory: str
    priority: str  # "P1" | "P2" | "P3" | "P4"
    confidence: float  # 0.0 to 1.0
    needs_review: bool  # True if confidence < CONFIDENCE_THRESHOLD
    reasoning: str
    # Multi-category support: secondary classification for tickets spanning 2 categories
    secondary_category: Optional[str] = None
    secondary_subcategory: Optional[str] = None
    secondary_confidence: Optional[float] = None  # used if > 0.4

@dataclass
class SimilarTicket:
    ticket: Ticket
    similarity_score: float
    matching_reason: str

@dataclass
class KnowledgeArticle:
    kb_id: str
    title: str
    category: str
    subcategory: str
    symptoms: list
    resolution_steps: list
    diagnostic_steps: list
    escalation_criteria: str
    relevance_score: float

@dataclass
class Evidence:
    source_id: str  # ticket ID or KB article ID (e.g., "KB-0018")
    source_type: str  # "ticket" or "knowledge_base"
    relevant_excerpt: str
    support_reason: str

@dataclass
class ResolutionRecommendation:
    recommended_resolution: str
    confidence: float
    evidence: list  # list[Evidence]
    alternative_resolutions: list  # list[str]
    needs_escalation: bool
    knowledge_articles_used: list  # list[str] — KB IDs used

@dataclass
class FeedbackRecord:
    ticket_id: str
    feedback_type: str  # "useful" | "incorrect" | "needs_escalation"
    classification_correct: Optional[bool] = None
    correct_category: Optional[str] = None
    correct_subcategory: Optional[str] = None
    retrieval_relevant: Optional[bool] = None
    resolution_helpful: Optional[bool] = None
    analyst_notes: Optional[str] = None
    missing_kb_topic: Optional[str] = None  # gap detection

@dataclass
class AnalysisResult:
    """Final output combining all stages"""
    input_ticket: Ticket
    classification: ClassificationResult
    similar_tickets: list  # list[SimilarTicket]
    knowledge_articles: list  # list[KnowledgeArticle]
    resolution: ResolutionRecommendation

# === CONSTANTS ===
CONFIDENCE_THRESHOLD = 0.7
TOP_K_SIMILAR = 5
TOP_K_KB = 3
VECTOR_DB_PATH = "data/processed/chroma_db"
TICKET_COLLECTION = "tickets"
KB_COLLECTION = "knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KB_PATH = "data/knowledge_base/articles.json"

# === ACTUAL DATASET TAXONOMY (18 subcategories from the .xlsx) ===
TAXONOMY = {
    "Database": ["DB2", "MS SQL Server", "Oracle"],
    "Hardware": ["CPU", "Disk", "Keyboard", "Memory", "Monitor", "Mouse"],
    "Network": ["DHCP", "DNS", "IP Address", "VPN", "Wireless"],
    "Software": ["Email", "Operating System"],
    "Inquiry/Help": ["Antivirus", "Internal Application"],
}

# === SUBCATEGORY → KB ARTICLE PRE-MAPPING ===
# Uses (Category, Subcategory) tuples because subcategory names alone can be ambiguous.
# Coverage: exact=4, broad=3, partial=7, unsupported=4
# "broad" means the subcategory maps to ALL articles in that macro category (e.g., DB2 → all 5 DB articles)
# "unsupported" means no KB article covers this — falls back to full vector search
SUBCATEGORY_TO_KB = {
    # Network — best coverage (exact matches)
    ("Network", "VPN"): ["KB-0018", "KB-0019"],           # exact
    ("Network", "Wireless"): ["KB-0020", "KB-0021"],       # exact (WiFi articles)
    ("Network", "DNS"): ["KB-0023"],                       # exact
    ("Network", "DHCP"): ["KB-0018", "KB-0023"],           # partial (network infra)
    ("Network", "IP Address"): ["KB-0023", "KB-0022"],     # partial (DNS + firewall)

    # Database — broad mapping (DB2/Oracle/MSSQL all map to all DB articles, ranked by text)
    ("Database", "DB2"): ["KB-0001", "KB-0002", "KB-0003", "KB-0004", "KB-0005"],       # broad
    ("Database", "MS SQL Server"): ["KB-0001", "KB-0002", "KB-0003", "KB-0004", "KB-0005"],  # broad
    ("Database", "Oracle"): ["KB-0001", "KB-0002", "KB-0003", "KB-0004", "KB-0005"],    # broad

    # Hardware — partial coverage
    ("Hardware", "Monitor"): ["KB-0008"],                  # exact
    ("Hardware", "Keyboard"): ["KB-0012"],                 # partial (KB has Keyboard/Mouse)
    ("Hardware", "Mouse"): ["KB-0012"],                    # partial (KB has Keyboard/Mouse)
    ("Hardware", "CPU"): ["KB-0006", "KB-0007"],           # partial (laptop performance articles)
    ("Hardware", "Disk"): ["KB-0006"],                     # partial (laptop hardware)
    ("Hardware", "Memory"): ["KB-0006", "KB-0007"],        # partial (laptop hardware)

    # Software — partial coverage
    ("Software", "Email"): ["KB-0029", "KB-0030"],         # partial (application error articles)
    ("Software", "Operating System"): ["KB-0026", "KB-0027", "KB-0028", "KB-0029", "KB-0030"],  # partial

    # Inquiry/Help — unsupported (no direct KB match, relies on vector fallback)
    ("Inquiry/Help", "Antivirus"): [],                     # unsupported → vector fallback
    ("Inquiry/Help", "Internal Application"): [],          # unsupported → vector fallback
}

# === LLM CLIENT HELPER ===
def get_llm_client():
    """Returns an OpenAI-compatible client configured for current provider."""
    from openai import OpenAI
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
```

---

## PERSON 1 — Data Pipeline + Knowledge Base + Embeddings + Retrieval

**Branch name:** `feat/data-pipeline`
**You own:** `data/`, `src/data_pipeline.py`, `src/kb_indexer.py`, `src/retriever.py`, `eval/retrieval_eval.py`

### Prompt for your AI agent:

```
You are helping me build the DATA PIPELINE, KNOWLEDGE BASE INDEXER, and RETRIEVAL components for an IT Ticket Analysis hackathon project.

CONTEXT: We're building an "Intelligent IT Ticket Analysis and Resolution Assistant" for HCL Tech's recruitment hackathon.

CRITICAL ARCHITECTURE DECISION: We use TWO-SOURCE RAG. The HuggingFace ticket dataset has NO resolution text, so we CANNOT use tickets for resolution. Instead:
  - Source 1: Ticket corpus (1800 tickets from HuggingFace) → used for similarity search + classification validation
  - Source 2: Knowledge Base (30 curated runbook articles in data/knowledge_base/articles.json) → used for resolution recommendations
Both are indexed in separate ChromaDB collections. This mirrors HCL's own production system which uses RAG over KEDB + runbooks.

I own the data ingestion, embedding, vector store, knowledge base indexing, and BOTH retrieval functions. Other teammates handle classification (Person 2), resolution synthesis (Person 3), and the UI (Person 4).

LLM SETUP: We use the `openai` Python package with Groq's API (free, OpenAI-compatible).
  - Set GROQ_API_KEY env var, or use LLM_PROVIDER=ollama for local dev with Ollama
  - Import from config: from src.config import get_llm_client, LLM_MODEL

REPO STRUCTURE — I work in these files:
- data/raw/                          — downloaded HuggingFace dataset
- data/processed/                    — cleaned data + ChromaDB persistent store
- data/knowledge_base/articles.json  — 30 curated KB articles (already generated — see below)
- data/sample_tickets.json           — 5+ demo tickets
- src/data_pipeline.py               — data loading, cleaning, embedding tickets
- src/kb_indexer.py                  — knowledge base → ChromaDB indexing
- src/retriever.py                   — TWO retrieval functions
- src/config.py                      — shared dataclasses (already defined, do not modify)
- eval/retrieval_eval.py             — retrieval quality metrics
- requirements.txt                   — add my deps

TASK 1 — Download and prepare the TICKET CORPUS:
1. Download "Noise144/ticket_classification_IT_EN" from HuggingFace (1800 IT tickets, 18 subcategories). Use the `datasets` library. If HuggingFace download fails, use the Dataset_ticket.xlsx file directly.

   IMPORTANT — ACTUAL DATASET FORMAT (discovered after inspection):
   - Columns: Category, Subcategory, Text (single text field — NO separate title/description)
   - 100 tickets per subcategory, 1800 total
   - Actual subcategories:
     * Database: DB2, MS SQL Server, Oracle
     * Hardware: CPU, Disk, Keyboard, Memory, Monitor, Mouse
     * Network: DHCP, DNS, IP Address, VPN, Wireless
     * Software: Email, Operating System
     * Inquiry/Help: Antivirus, Internal Application

2. Clean and normalize: strip whitespace, standardize category names, remove duplicates.
3. DO NOT generate fake resolutions. The resolution field stays null. Tickets are for SIMILARITY SEARCH ONLY.
4. Map to the Ticket dataclass. Since the dataset has a single `Text` field (no separate title):
   - title = first sentence of Text (or first 100 chars)
   - description = full Text
   - Save as data/processed/tickets.json
5. Build the TICKET embedding index in ChromaDB:
   - Collection name: "tickets"
   - Use "all-MiniLM-L6-v2" for embeddings (same model used for KB indexing).
   - FEATURE ENGINEERING: Create structured embedding input:
     f"Category: {category}. Subcategory: {subcategory}. {text}"
   - Store metadata: {category, subcategory, has_resolution: false}
   - Persist to data/processed/chroma_db/

TASK 2 — Index the KNOWLEDGE BASE:
1. The KB articles are already in data/knowledge_base/articles.json (30 articles, generated separately).
   Each article has: kb_id, title, category, subcategory, symptoms, common_error_codes, root_cause, diagnostic_steps, resolution (with steps, estimated_time, requires_admin), prevention, escalation_criteria, related_kb_ids, tags, priority_hint, last_updated.
   NOTE: The KB articles use the ORIGINAL taxonomy (e.g., "DB performance", "VPN"), which doesn't exactly match the dataset taxonomy (e.g., "DB2", "Wireless"). The SUBCATEGORY_TO_KB crosswalk in config.py handles this mapping.
2. In src/kb_indexer.py, implement:
   - load_knowledge_base() -> list[dict]
   - index_knowledge_base(articles: list[dict]) -> None
3. Use the SAME embedding model as tickets: "all-MiniLM-L6-v2"
   Both indexes MUST use the same model for consistent similarity scores.
4. For embedding, create a rich text combining multiple fields:
   f"{title}. Symptoms: {'. '.join(symptoms)}. Root cause: {root_cause}. Resolution: {'. '.join(resolution['steps'])}"
5. ChromaDB collection name: "knowledge_base"
6. Store metadata: {category, subcategory, priority_hint, requires_admin, title, tags (comma-joined)}
7. Use collection.upsert() instead of add() so re-running is safe.

TASK 3 — Build the DUAL RETRIEVER:
1. In src/retriever.py, implement TWO functions:

   a. find_similar_tickets(ticket: Ticket, top_k: int = 5) -> list[SimilarTicket]
      - Embeds the input ticket's title+description (or just description if title is empty)
      - Uses "all-MiniLM-L6-v2" for embeddings
      - Queries the "tickets" ChromaDB collection
      - Returns ranked similar tickets with scores and matching reasons
      - Support optional category filter (use ChromaDB where clause) when classification confidence > 0.8
      - NOTE: The retriever should also accept raw dataset row format {Category, Subcategory, Text} as input, not just Ticket objects — add a helper to convert

   b. find_knowledge_articles(ticket: Ticket, classification: ClassificationResult = None, top_k: int = 3) -> list[KnowledgeArticle]
      TWO-SPEED RETRIEVAL (fast path + vector fallback):
      
      FAST PATH (when classification confidence > 0.7):
      - Look up SUBCATEGORY_TO_KB[classification.subcategory] from config
      - Directly fetch those KB articles by ID from ChromaDB (no embedding, no search)
      - This is a dict lookup + ID fetch — essentially instant
      - If secondary_category exists (confidence > 0.4), also fetch its mapped articles
      - Merge, deduplicate by kb_id
      - Set relevance_score = 1.0 for direct matches (they're pre-mapped to this subcategory)
      
      VECTOR FALLBACK (when classification confidence <= 0.7 OR fast path returns nothing):
      - Embeds the input ticket's title+description
      - Queries the "knowledge_base" ChromaDB collection via vector similarity
      - This catches edge cases and ambiguous tickets that the mapping might miss
      - relevance_score = 1 - distance (cosine)
      
      HYBRID (best of both — use this for demo):
      - Run fast path AND vector search
      - Merge results: fast-path articles get a relevance boost (+0.1)
      - Deduplicate by kb_id, keep highest score
      - Return top_k results
      
      Returns KnowledgeArticle objects (import from src.config)

2. Import all dataclasses from src.config.
3. Person 3 will call BOTH functions from the pipeline.

TASK 4 — Create sample tickets for demo:
1. Create data/sample_tickets.json with at least 6 realistic IT tickets that DON'T exist in the training data.
   - A VPN connectivity issue: "Can't connect to corporate VPN from home office"
   - A database slow query: "Oracle database queries running very slow, reports taking 10+ minutes"
   - A monitor problem: "External monitor goes black intermittently after Windows update"
   - An email issue: "Outlook keeps crashing when opening attachments"
   - A multi-category ticket: "VPN keeps dropping and laptop overheating at the same time"
   - A near-duplicate of an existing ticket (to test duplicate detection)
2. These will be used by Person 4 for demo and Person 3 for worked examples.

TASK 5 — Retrieval evaluation:
1. In eval/retrieval_eval.py:
   - For each sample ticket, run BOTH retrieval functions
   - For ticket retrieval: check if top-5 results are from the same category. Compute precision@k (k=1,3,5) and MRR.
   - For KB retrieval: check if the top-3 articles are from the matching subcategory.
   - Save results to eval/results/retrieval_metrics.json.

DEPENDENCIES to add to requirements.txt:
sentence-transformers, chromadb, datasets, numpy

IMPORTANT:
- Use the exact dataclass types from src/config.py. Do not change the interface.
- The TWO retriever functions must be importable as:
    from src.retriever import find_similar_tickets, find_knowledge_articles
- Person 3 will call BOTH functions from the resolver.
- Run: python -m src.kb_indexer to index the knowledge base separately from tickets.
- Commit often. Push to branch feat/data-pipeline.
```

---

## PERSON 2 — Classification + Confidence Engine

**Branch name:** `feat/classifier`
**You own:** `src/classifier.py`, `eval/classification_eval.py`

### Prompt for your AI agent:

```
You are helping me build the TICKET CLASSIFICATION component for an IT Ticket Analysis hackathon project.

CONTEXT: We're building an "Intelligent IT Ticket Analysis and Resolution Assistant" for HCL Tech's recruitment hackathon. I own ticket classification — assigning category, subcategory, priority, and confidence scores. Other teammates handle data/retrieval (Person 1), resolution recommendation (Person 3), and the UI (Person 4).

LLM SETUP: We use the `openai` Python package with Groq's free API (OpenAI-compatible).
  - DO NOT use "gpt-3.5-turbo" or any OpenAI model. We don't have OpenAI API keys.
  - Import from config: from src.config import get_llm_client, LLM_MODEL
  - Usage:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,  # "llama-3.1-70b-versatile" on Groq
        messages=[...],
        temperature=0.1,
        response_format={"type": "json_object"}  # for structured output
    )
  - For local dev: set LLM_PROVIDER=ollama to use local Ollama with qwen2.5:7b

REPO STRUCTURE — I work in these files:
- src/classifier.py              — classification engine
- src/config.py                  — shared dataclasses (already defined by team, do not modify)
- eval/classification_eval.py    — classification metrics and confusion matrix
- requirements.txt               — add my deps

SHARED INTERFACE (from src/config.py — DO NOT MODIFY):
@dataclass
class Ticket:
    id, title, description, category, subcategory, priority, status, resolution, metadata

@dataclass
class ClassificationResult:
    category, subcategory, priority (P1-P4), confidence (0-1), needs_review, reasoning

CONFIDENCE_THRESHOLD = 0.7

TASK 1 — Build the classifier:
1. In src/classifier.py, implement:
   classify_ticket(ticket: Ticket) -> ClassificationResult

2. Use the LLM with structured JSON output. The prompt should:
   a. Take the ticket title and description
   b. Extract signals: error codes (regex patterns), urgency keywords ("urgent", "ASAP", "outage"), affected user count, system names
   c. Return category, subcategory, priority with confidence
   d. Use this exact taxonomy (18 subcategories — from the ACTUAL dataset, not HuggingFace description):
      * Database: DB2, MS SQL Server, Oracle
      * Hardware: CPU, Disk, Keyboard, Memory, Monitor, Mouse
      * Network: DHCP, DNS, IP Address, VPN, Wireless
      * Software: Email, Operating System
      * Inquiry/Help: Antivirus, Internal Application
   e. Priority mapping:
      * P1: Complete outage, many users affected, security breach
      * P2: Major degradation, workaround exists but painful
      * P3: Minor issue, single user affected
      * P4: Cosmetic, informational, nice-to-have
   f. Confidence should reflect how clearly the ticket maps to one category. Ambiguous tickets (e.g., could be Network or Software) get lower confidence.
   g. reasoning: 1-2 sentences explaining WHY this category was chosen, citing specific phrases from the ticket.

3. Set needs_review = True when confidence < CONFIDENCE_THRESHOLD (0.7).

4. MULTI-CATEGORY CLASSIFICATION:
   - Some tickets span two categories (e.g., "VPN not working and laptop keeps crashing" = Network + Hardware).
   - The LLM prompt should ask: "Is there a SECONDARY category? If the ticket clearly involves a second distinct issue, provide secondary_category, secondary_subcategory, and secondary_confidence."
   - Only populate secondary fields if secondary_confidence > 0.4 (not noise).
   - The keyword fallback should also check for multi-category: if keywords from 2 different categories are found, set the secondary.
   - Example LLM prompt addition:
     ```
     If this ticket involves TWO distinct issues from different categories,
     also provide:
     - secondary_category, secondary_subcategory, secondary_confidence (0-1)
     Only if the second issue is clearly present (confidence > 0.4).
     If the ticket is about one issue, set these to null.
     ```

TASK 2 — Signal extraction (FEATURE ENGINEERING):
Before sending to LLM, extract structured signals from the ticket text:
   - Error codes: regex for patterns like "0x[A-F0-9]+", "ERR-\d+", "ORA-\d+", HTTP status codes
   - Urgency markers: "urgent", "critical", "ASAP", "down", "outage", "affecting X users"
   - System mentions: map specific keywords to categories (e.g., "VPN" → Network, "SQL" → Database)
   - User impact: try to extract number of affected users
Include these extracted signals in the LLM prompt for better classification accuracy.

TASK 3 — Fallback strategy (3-layer resilience):
1. If the LLM call fails (timeout, rate limit, network error), fall back to keyword-based classifier.
2. Implement classify_ticket_keyword_fallback(ticket: Ticket) -> ClassificationResult
3. Keyword matching rules (build a dict mapping keywords to ACTUAL subcategories):
   Build a dict mapping keywords to the ACTUAL subcategories:
   - "VPN", "tunnel", "remote access" → Network/VPN
   - "wireless", "WiFi", "SSID" → Network/Wireless
   - "DNS", "resolve", "nslookup" → Network/DNS
   - "DHCP", "lease" → Network/DHCP
   - "IP", "IP address", "subnet" → Network/IP Address
   - "DB2", "db2" → Database/DB2
   - "SQL Server", "MSSQL", "Microsoft SQL" → Database/MS SQL Server
   - "Oracle", "ORA-" → Database/Oracle
   - "CPU", "processor", "slow", "performance" → Hardware/CPU
   - "disk", "storage", "hard drive", "SSD" → Hardware/Disk
   - "keyboard" → Hardware/Keyboard
   - "memory", "RAM" → Hardware/Memory
   - "monitor", "display", "screen" → Hardware/Monitor
   - "mouse" → Hardware/Mouse
   - "email", "outlook", "mail" → Software/Email
   - "OS", "Windows", "operating system", "boot" → Software/Operating System
   - "antivirus", "virus", "malware", "security" → Inquiry/Help/Antivirus
   - "application", "app", "internal" → Inquiry/Help/Internal Application
   - Default fallback: Inquiry/Help / Internal Application
4. Fallback results: cap confidence at 0.5, set needs_review=True, reasoning = "Classified using keyword fallback due to LLM unavailability".
5. IMPORTANT: The fallback must NEVER crash. Wrap LLM call in try/except, always return a ClassificationResult.
6. If BOTH Groq and Ollama are unavailable, the keyword fallback is the safety net — the system keeps working.

TASK 4 — Evaluation:
1. In eval/classification_eval.py:
   - Load the processed dataset from data/processed/tickets.json (Person 1 creates this).
   - Run classify_ticket() on a sample of 50-100 tickets with ground truth labels.
   - Compute: overall accuracy, per-class precision/recall/F1, confusion matrix.
   - Save confusion matrix image: eval/results/confusion_matrix.png
   - Save metrics JSON: eval/results/classification_metrics.json
   - NOTE: Use Groq API with rate limiting (add 0.5s delay between calls) to avoid hitting free tier limits.

TASK 5 — Batch classification utility:
1. classify_batch(tickets: list[Ticket]) -> list[ClassificationResult]
2. Handle rate limiting (delays between calls) and auto-fallback on failure.
3. Track success rate and fallback rate for reporting.

DEPENDENCIES to add to requirements.txt:
openai, scikit-learn, matplotlib, seaborn

IMPORTANT:
- DO NOT import or use the openai.OpenAI constructor directly. Use: from src.config import get_llm_client, LLM_MODEL
- The function signature must be exactly: classify_ticket(ticket: Ticket) -> ClassificationResult
- Person 3 will call classify_ticket() from the pipeline — make sure it works standalone.
- Person 4 will display the classification result including the confidence badge and reasoning.
- If no API key is available during development, add a MOCK_MODE that returns plausible results for UI testing.
- Commit often. Push to branch feat/classifier.
```

---

## PERSON 3 — Resolution Recommendation + Pipeline Orchestration

**Branch name:** `feat/resolver`
**You own:** `src/resolver.py`, `src/pipeline.py`, `docs/architecture.md`, `docs/worked_examples.md`

### Prompt for your AI agent:

```
You are helping me build the RESOLUTION RECOMMENDATION ENGINE and the ORCHESTRATION PIPELINE for an IT Ticket Analysis hackathon project.

CONTEXT: We're building an "Intelligent IT Ticket Analysis and Resolution Assistant" for HCL Tech's recruitment hackathon. I own the resolution recommendation and the main pipeline that connects classification → retrieval → resolution. Other teammates handle data/retrieval (Person 1), classification (Person 2), and the UI (Person 4).

CRITICAL ARCHITECTURE — TWO-SOURCE RAG:
The HuggingFace dataset has NO resolution text. We use two retrieval sources:
  - Source 1: Similar TICKETS (from Person 1's find_similar_tickets) → provides pattern matching and context, but NO resolutions
  - Source 2: Knowledge Base ARTICLES (from Person 1's find_knowledge_articles) → provides actual resolution procedures, diagnostic steps, escalation criteria
The resolver SYNTHESIZES a recommendation by combining KB article procedures with ticket context. This is the core RAG pattern — evidence-backed resolution from curated runbooks.

This mirrors HCL's own production system: RAG over KEDB + runbooks + ticket history.

LLM SETUP: We use the `openai` Python package with Groq's free API (OpenAI-compatible).
  - Import from config: from src.config import get_llm_client, LLM_MODEL
  - DO NOT use "gpt-3.5-turbo". We use Groq's llama-3.1-70b-versatile.
  - For local dev: set LLM_PROVIDER=ollama

REPO STRUCTURE — I work in these files:
- src/resolver.py         — resolution recommendation engine
- src/pipeline.py         — orchestrates the full flow
- src/config.py           — shared dataclasses (already defined, do not modify)
- docs/architecture.md    — system architecture doc
- docs/worked_examples.md — 3+ worked examples for submission
- requirements.txt        — add my deps

KEY DATACLASSES (from src/config.py — DO NOT MODIFY):
- Ticket, ClassificationResult, SimilarTicket, KnowledgeArticle — inputs
- Evidence (source_id, source_type ["ticket"|"knowledge_base"], relevant_excerpt, support_reason)
- ResolutionRecommendation (recommended_resolution, confidence, evidence, alternative_resolutions, needs_escalation, knowledge_articles_used)
- FeedbackRecord (for feedback capture)
- AnalysisResult (input_ticket, classification, similar_tickets, knowledge_articles, resolution)

TASK 1 — Build the resolver:
1. In src/resolver.py, implement:
   recommend_resolution(
       ticket: Ticket,
       classification: ClassificationResult,
       similar_tickets: list[SimilarTicket],
       knowledge_articles: list[KnowledgeArticle]    # <-- NEW: from KB retrieval
   ) -> ResolutionRecommendation

2. The LLM prompt for resolution synthesis should include:
   a. The input ticket (title + description)
   b. The classification result (category, subcategory, priority)
   c. Similar tickets (for context — what similar problems looked like)
   d. Knowledge base articles (THE PRIMARY RESOLUTION SOURCE):
      - Include each article's: title, diagnostic_steps, resolution_steps, escalation_criteria
   e. Ask the LLM to:
      - Synthesize a step-by-step resolution adapted to THIS specific ticket
      - Cite which KB article(s) each step comes from
      - Suggest alternatives from other matching KB articles
      - Determine if escalation is needed based on KB escalation_criteria
      - MULTI-CATEGORY: if classification has a secondary_category, the LLM prompt should say:
        "This ticket involves two issues: [primary] and [secondary]. Provide a combined resolution
        that addresses both. Order steps logically — resolve the blocking issue first (e.g., fix
        network before troubleshooting the app that needs network). Clearly label which steps
        address which issue."

3. Evidence traceability — CRITICAL for the hackathon:
   - Every resolution step must cite its source as an Evidence object
   - source_type = "knowledge_base" for KB-derived steps, "ticket" for pattern-matched insights
   - source_id = the KB-ID (e.g., "KB-0018") or ticket ID
   - relevant_excerpt = the specific step/text from the source
   - This is what makes our system auditable and enterprise-ready

4. Escalation logic (needs_escalation = True when):
   - No matching KB articles found (relevance_score < 0.3 for all)
   - Classification confidence < 0.5
   - Ticket mentions: security breach, data loss, multiple users affected, compliance
   - KB article's escalation_criteria matches the ticket scenario

5. Populate knowledge_articles_used with the KB-IDs used in the recommendation.

6. KNOWLEDGE GAP HANDLING — when no KB article matches:
   - If all KB articles have relevance_score < 0.3:
     - Set needs_escalation = True
     - recommended_resolution = "No matching knowledge base article found. This appears to be a novel issue type. Recommended: escalate to L2/L3 support for manual investigation."
     - confidence = 0.1
     - evidence = [] (empty — we won't fabricate sources)
     - The LLM can still provide directional guidance based on classification + similar ticket patterns, but must clearly label it as "suggested, not KB-backed"
   - Log the gap: save {ticket_id, classification, "no_kb_match"} so the team knows which topics need new KB articles

7. LLM-DOWN FALLBACK for resolver:
   - If the LLM call fails, DO NOT return empty results.
   - Instead, return the TOP KB article's resolution_steps directly (no synthesis, just the raw steps).
   - Set reasoning = "Resolution sourced directly from knowledge base article {kb_id} without LLM synthesis (LLM unavailable)."
   - This still gives the analyst actionable steps — just not customized to the specific ticket.
   - The UI should indicate this is a "raw KB match" vs a "synthesized resolution".

TASK 2 — Build the orchestration pipeline:
1. In src/pipeline.py, implement:
   analyze_ticket(ticket: Ticket) -> AnalysisResult
   This is the MAIN ENTRY POINT that Person 4's UI calls.

2. The flow is:
   a. classify_ticket(ticket) → ClassificationResult          [from src.classifier]
   b. find_similar_tickets(ticket, top_k=5) → list[SimilarTicket]  [from src.retriever]
   c. find_knowledge_articles(ticket, classification, top_k=3) → list[KnowledgeArticle]  [from src.retriever]
   d. recommend_resolution(ticket, classification, similar_tickets, knowledge_articles) → ResolutionRecommendation  [from src.resolver]
   e. Return AnalysisResult combining all of the above

3. Error handling — GRACEFUL DEGRADATION (system never crashes, always returns something useful):
   - If classification fails: use keyword fallback from Person 2's classify_ticket_keyword_fallback(). If that also fails, set category="Unknown", confidence=0.0, needs_review=True.
   - If ticket retrieval fails: continue with KB retrieval only. Similar tickets = [].
   - If KB retrieval fails: continue with similar tickets only. Knowledge articles = [].
   - If KB retrieval returns empty/low relevance (all < 0.3): set needs_escalation=True, return "No matching knowledge base articles. Novel issue — escalate to L2/L3."
   - If LLM resolution synthesis fails: return the top KB article's raw resolution_steps directly without synthesis. Mark as "raw KB match".
   - If EVERYTHING fails (no LLM, no vector DB): return a minimal AnalysisResult with needs_escalation=True and "System degraded — all components unavailable. Please escalate to IT support manually."
   - Log each step's success/failure for debugging. Track which fallback path was used.

4. Add timing: measure and log how long each step takes. Target: < 10 seconds total (HCL's benchmark).
   - Log which KB retrieval path was used: "fast_path" (subcategory lookup), "vector_search" (embedding similarity), or "hybrid" (both merged)
   - Include timing per step in AnalysisResult.metadata or logs so Person 4 can display it:
     e.g., {"classification_ms": 800, "ticket_retrieval_ms": 120, "kb_retrieval_ms": 15, "resolution_ms": 1200, "total_ms": 2135, "kb_retrieval_mode": "fast_path"}
   - The fast path should bring KB retrieval from ~200ms down to ~10ms — mention this in the presentation as an optimization

TASK 3 — Feedback capture:
1. In src/pipeline.py, implement:
   record_feedback(feedback: FeedbackRecord) -> None
   - Saves feedback to data/processed/feedback.json (append to list)
   - This captures 4 improvement signals:
     a. Classification correction: was the category/subcategory right?
     b. Retrieval relevance: were the similar tickets and KB articles relevant?
     c. Resolution quality: was the recommendation helpful?
     d. KB gap detection: if missing_kb_topic is set, log it as a content gap

TASK 4 — Worked examples documentation:
1. In docs/worked_examples.md, create 3 detailed worked examples showing the FULL pipeline:
   - Input ticket
   - Classification output (category, subcategory, priority, confidence)
   - Similar tickets found (top 3, with scores)
   - Knowledge articles matched (top 2-3, with relevance scores)
   - Synthesized resolution with evidence citations
   - Show the traceability chain: recommendation → KB article → specific step
2. Use sample tickets from data/sample_tickets.json.

TASK 5 — Architecture documentation:
1. In docs/architecture.md, document:
   - 6-layer architecture: Input → Classification → Two-Source Retrieval → Resolution Synthesis → Analyst Review → Feedback Capture
   - Data flow diagram (mermaid)
   - TWO-SOURCE RAG explanation: why tickets for similarity, KB for resolution
   - Comparison with HCL's production system (25K monthly tickets, <10s response, 80%+ positive feedback)
   - Technology choices: Groq (LLM), ChromaDB (vector store), sentence-transformers (embeddings), Streamlit (UI)
   - Feature engineering: structured embeddings, signal extraction, metadata filtering

TASK 6 — Mock pipeline (PUSH FIRST):
1. Implement mock_analyze_ticket(ticket: Ticket) -> AnalysisResult
   - Returns realistic hardcoded results INCLUDING knowledge_articles
   - Person 4 needs this within the first 30 minutes to build UI
2. In pipeline.py, use MOCK_MODE env var to toggle between mock and real.

DEPENDENCIES to add to requirements.txt:
openai (for Groq — same package, different base_url)

IMPORTANT:
- Push the mock pipeline (Task 6) FIRST so Person 4 can start building UI immediately.
- Import from teammates' modules:
    from src.classifier import classify_ticket
    from src.retriever import find_similar_tickets, find_knowledge_articles
- If their modules aren't ready yet, catch ImportError and fall back to mock.
- The function analyze_ticket(ticket: Ticket) -> AnalysisResult is the ONLY function Person 4 calls.
- Commit often. Push to branch feat/resolver.
```

---

## PERSON 4 — Frontend (Streamlit) + Presentation + README

**Branch name:** `feat/frontend`
**You own:** `app/`, `README.md`, `presentation/`, overall polish

### Prompt for your AI agent:

```
You are helping me build the FRONTEND and PRESENTATION for an IT Ticket Analysis hackathon project.

CONTEXT: We're building an "Intelligent IT Ticket Analysis and Resolution Assistant" for HCL Tech's recruitment hackathon. I own the Streamlit UI, the README, and the presentation. Other teammates handle data/retrieval (Person 1), classification (Person 2), and resolution/pipeline (Person 3).

KEY ARCHITECTURE POINTS (for README and presentation):
- TWO-SOURCE RAG: Ticket corpus for similarity matching + Knowledge Base (30 curated runbook articles) for resolution. The HuggingFace dataset has no resolution text, so we built a separate KB — this mirrors HCL's own production system.
- LLM: Groq API (llama-3.1-70b-versatile) — free, fast, no OpenAI needed
- Embeddings: sentence-transformers (all-MiniLM-L6-v2) + ChromaDB
- HCL Benchmark: Their production system handles 25K+ monthly tickets with <10s response time and 80%+ positive feedback. Our prototype targets the same architecture at smaller scale.

REPO STRUCTURE — I work in these files:
- app/app.py               — main Streamlit application
- app/components/           — UI component modules
- README.md                — setup + run instructions + architecture summary
- presentation/slides.md   — presentation outline/script
- requirements.txt         — add my deps

THE ONLY FUNCTION I CALL FROM BACKEND:
```python
from src.pipeline import analyze_ticket
from src.config import Ticket, AnalysisResult, FeedbackRecord

# Create a Ticket from user input
ticket = Ticket(id="new-1", title="...", description="...")

# Get the full analysis
result: AnalysisResult = analyze_ticket(ticket)
# result.classification       → ClassificationResult
# result.similar_tickets      → list[SimilarTicket]
# result.knowledge_articles   → list[KnowledgeArticle]   ← NEW in two-source RAG
# result.resolution           → ResolutionRecommendation

# Record feedback
from src.pipeline import record_feedback
record_feedback(FeedbackRecord(ticket_id="new-1", feedback_type="useful", ...))
```

The AnalysisResult contains:
- classification: {category, subcategory, priority, confidence, needs_review, reasoning}
- similar_tickets: [{ticket, similarity_score, matching_reason}, ...]
- knowledge_articles: [{kb_id, title, category, subcategory, symptoms, resolution_steps, diagnostic_steps, escalation_criteria, relevance_score}, ...]
- resolution: {recommended_resolution, confidence, evidence: [{source_id, source_type, relevant_excerpt, support_reason}], alternative_resolutions, needs_escalation, knowledge_articles_used}

TASK 1 — Build the Streamlit app (START IMMEDIATELY with mock data):
1. Don't wait for backend. Create a mock_analyze_ticket function that returns hardcoded realistic results (including knowledge_articles list) so you can build the full UI right now.
2. Use MOCK_MODE env var to toggle: if os.getenv("MOCK_MODE", "true") == "true": use mock else: from src.pipeline import analyze_ticket

TASK 2 — UI Layout:
Build a clean, professional Streamlit app:

PAGE CONFIG:
- st.set_page_config(page_title="IT Ticket Assistant", page_icon="🎫", layout="wide")

HEADER:
- "🎫 Intelligent IT Ticket Analysis and Resolution Assistant"
- Subtitle: "Two-Source RAG — Evidence-backed resolutions from curated knowledge base"
- HCL Tech branding colors (primary blue: #0073e6)

SIDEBAR:
- "Sample Tickets" dropdown to load pre-built demo tickets
- Filters: category, priority dropdowns
- "About" section: brief system description
- "Architecture" section: mention Two-Source RAG, Groq LLM, ChromaDB

MAIN AREA:

Section 1 — Input
- Text input for ticket title
- Text area for ticket description
- "🔍 Analyze Ticket" button (primary, large)

Section 2 — Classification Result
- Card with:
  - PRIMARY: Category + Subcategory badges (colored pills)
  - Priority badge (P1=🔴red, P2=🟠orange, P3=🟡yellow, P4=🟢green)
  - Confidence meter (progress bar: green >0.8, yellow 0.5-0.8, red <0.5)
  - SECONDARY (if present): show a second row of badges with its own confidence bar
    - Label: "Secondary Issue Detected" with category + subcategory badges
    - Only shown when secondary_confidence > 0.4
    - This impresses judges — shows the system handles multi-faceted tickets
  - ⚠️ "Low confidence — recommend analyst review" warning if needs_review
  - Reasoning text in italics

Section 3 — Knowledge Base Matches (NEW — this is key differentiator)
- Show top 3 matched KB articles:
  - Each article: KB-ID badge, Title, Category, Relevance Score
  - Expandable sections showing:
    - Symptoms list
    - Diagnostic Steps (numbered)
    - Resolution Steps (numbered)
    - Escalation Criteria
- This shows the analyst WHERE the resolution came from — enterprise traceability

Section 4 — Similar Tickets
- Table of top 5 similar tickets:
  - Ticket ID, Title, Similarity Score (as %), Matching Reason
  - Color-coded: >80% green, 50-80% yellow, <50% grey
  - Expandable detail for each

Section 5 — Recommended Resolution (HERO SECTION — make prominent)
- Large card:
  - "📋 Recommended Resolution" header
  - Resolution text (well-formatted, step-by-step)
  - Confidence indicator
  - Resolution source indicator:
    - 🟢 "LLM-Synthesized" — full RAG synthesis worked
    - 🟡 "Raw KB Match" — LLM was unavailable, showing KB article steps directly
    - 🔴 "No KB Match — Escalation Required" — novel issue, no matching articles
  - 🚨 "Escalation Recommended" banner if needs_escalation
- EVIDENCE TRAIL (critical for hackathon):
  - For each Evidence item:
    - Source badge: "KB-0018" (blue) or "TKT-1234" (grey) showing source_type
    - Quoted excerpt in blockquote style
    - Support reason explaining the connection
  - If evidence is empty (knowledge gap): show "⚠️ No evidence sources matched. This is a novel issue not covered by the knowledge base."
  - Label: "Traceability: every recommendation links back to its source"
- Alternative resolutions in collapsible section
- KB Articles Used: list of KB-IDs with titles

Section 5b — System Status Indicators (small bar below resolution)
- Show which components succeeded/failed:
  - ✅ Classification | ✅ Ticket Retrieval | ✅ KB Retrieval | ✅ LLM Synthesis
  - or: ✅ Classification | ✅ Ticket Retrieval | ⚠️ KB Retrieval (low match) | ❌ LLM (using fallback)
- This shows judges the system degrades gracefully rather than crashing

Section 6 — Analyst Feedback (4 improvement loops)
- Three action buttons: ✅ "Useful" | ❌ "Incorrect" | ⏫ "Needs Escalation"
- Expandable "Detailed Feedback" section:
  - "Was the classification correct?" Yes/No + correct category dropdown
  - "Were similar tickets relevant?" Yes/No
  - "Was the resolution helpful?" Yes/No
  - "Missing topic in knowledge base?" text field (KB gap detection)
  - Free-text notes field
- On submit: call record_feedback() and show success toast

Section 7 — Performance & Audit Log (collapsible)
- Processing time breakdown bar:
  - Classification: Xms | Ticket Retrieval: Xms | KB Retrieval: Xms (fast_path/vector) | Resolution: Xms | Total: Xms
  - Show as a horizontal stacked bar chart — visually demonstrates <10s target
  - Highlight KB retrieval mode: "⚡ Fast Path (15ms)" vs "🔍 Vector Search (200ms)"
- JSON viewer of complete AnalysisResult
- "📥 Export as JSON" download button

TASK 3 — Styling:
- Custom CSS for enterprise look: primary blue #0073e6, clean white cards with subtle shadows
- Use st.columns for responsive layout
- Make evidence trail visually clear — this is what differentiates us

TASK 4 — README.md:
Write a professional README:
- Project title + one-liner
- Architecture diagram (mermaid): show the two-source RAG flow
- Key innovation: "Two-Source RAG separating ticket similarity from knowledge-based resolution"
- Setup instructions: git clone, pip install -r requirements.txt, set GROQ_API_KEY, python -m src.kb_indexer, streamlit run app/app.py
- Tech stack: Python, Streamlit, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), Groq (LLM), NO OpenAI needed
- HCL alignment: mention our architecture mirrors HCL's production ITSM GenAI system
- Team members section
- Evaluation metrics section (placeholder for Person 2's numbers)

TASK 5 — Presentation outline (5 minutes):
In presentation/slides.md:
1. Problem (30s): "Enterprise IT teams handle 25K+ tickets/month. Manual triage takes 15-20 min per ticket. Resolution knowledge is scattered across wikis, runbooks, and tribal knowledge."
2. Our Solution (30s): "Two-Source RAG: ticket similarity for context + curated knowledge base for resolution. Evidence-backed, auditable, no hallucinated fixes."
3. Live Demo (2.5min): Walk through TWO tickets from different categories showing: classification → KB match → resolution with evidence trail → feedback capture.
4. Architecture (30s): 6-layer diagram. Highlight: subcategory-to-KB crosswalk, 3-layer fallback, knowledge gap detection. Compare with HCL's own production system.
5. Results (30s): Classification accuracy, retrieval precision, response time. Target: <10s (matching HCL benchmark).
6. Enterprise Readiness (30s): Audit trail, confidence thresholds, human-in-the-loop feedback, KB gap detection for continuous improvement, escalation for novel issues.

DEPENDENCIES to add to requirements.txt:
streamlit

IMPORTANT:
- START WITH MOCK DATA. Don't block on backend.
- analyze_ticket(ticket) → AnalysisResult is the ONLY backend call.
- The demo flow: pick sample ticket → click Analyze → see classification → KB matches → resolution with evidence → feedback. Should work in 2-3 seconds.
- EMPHASIZE the evidence trail in the UI — this is what makes it enterprise-grade vs a chatbot.
- Commit often. Push to branch feat/frontend.
```

---

## MERGE PLAN

### Checkpoint 1 (Hour 1-2):
```bash
# Person 3 merges first (has config.py + mock pipeline)
git checkout main && git merge feat/resolver

# Person 1 merges next (data + KB + retriever)
git merge feat/data-pipeline

# Person 2 merges (classifier)
git merge feat/classifier

# Person 4 merges last (frontend)
git merge feat/frontend
```

### Checkpoint 2 (Hour 4-5):
- Set GROQ_API_KEY and run: `python -m src.data_pipeline && python -m src.kb_indexer`
- Verify end-to-end: `streamlit run app/app.py` with real backend
- Fix integration issues
- Toggle MOCK_MODE=false

### Final (Hour 6-7):
- Run eval scripts, capture metrics
- Generate worked examples with real pipeline output
- Final README + presentation polish
- Tag release: `git tag v1.0-submission`

---

## FIRST 15 MINUTES — EVERYONE DOES THIS:

1. One person gets a Groq API key at https://console.groq.com (free, instant signup)
2. Share the key with the team (set as GROQ_API_KEY env var)
3. Person 3 pushes `src/config.py` with all dataclasses + LLM config to main FIRST
4. Person 3 also pushes the mock pipeline immediately
5. Everyone pulls main, creates their branch, starts working
6. Person 1: also download `data/knowledge_base/articles.json` (the 30 KB articles already generated)

The `config.py` and mock pipeline must be on main BEFORE anyone branches. This is the contract.

## ENVIRONMENT SETUP

```bash
# Everyone runs this:
pip install openai sentence-transformers chromadb datasets streamlit scikit-learn matplotlib seaborn numpy

# The embedding model (~80MB) will auto-download on first use:
# all-MiniLM-L6-v2

# Set the Groq API key:
export GROQ_API_KEY="gsk_..."

# For local dev with Ollama instead:
# ollama pull qwen2.5:7b
# export LLM_PROVIDER=ollama

# Verify LLM works:
python -c "
from src.config import get_llm_client, LLM_MODEL
client = get_llm_client()
r = client.chat.completions.create(model=LLM_MODEL, messages=[{'role':'user','content':'Say hello'}])
print(r.choices[0].message.content)
"
```
