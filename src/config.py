import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Auto-load .env if present
_env_path = Path(__file__).resolve().parents[1] / ".env"
if _env_path.is_file():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

# === LLM PROVIDER CONFIG — Switch via environment variable ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" or "ollama"

if LLM_PROVIDER == "ollama":
    LLM_MODEL = "qwen2.5:7b"
    LLM_BASE_URL = "http://localhost:11434/v1"
    LLM_API_KEY = "ollama"  # Ollama ignores keys but openai package requires one
else:  # groq (default)
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
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
    retrieval_source: str = "vector"  # "subcategory_map" | "vector_fallback" | "vector"

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
# Multilingual model — supports both ticket corpus and English KB
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
KB_PATH = "data/knowledge_base/articles.json"
KB_MAPPING_PATH = "data/knowledge_base/subcategory_to_kb.json"

# === ACTUAL DATASET TAXONOMY (18 subcategories from the .xlsx) ===
TAXONOMY = {
    "Database": ["DB2", "MS SQL Server", "Oracle"],
    "Hardware": ["CPU", "Disk", "Keyboard", "Memory", "Monitor", "Mouse"],
    "Network": ["DHCP", "DNS", "IP Address", "VPN", "Wireless"],
    "Software": ["Email", "Operating System"],
    "Inquiry/Help": ["Antivirus", "Internal Application"],
}

# === SUBCATEGORY → KB ARTICLE PRE-MAPPING ===
SUBCATEGORY_TO_KB = {
    ("Network", "VPN"): ["KB-0018", "KB-0019"],
    ("Network", "Wireless"): ["KB-0020", "KB-0021"],
    ("Network", "DNS"): ["KB-0023"],
    ("Network", "DHCP"): [],
    ("Network", "IP Address"): [],
    ("Database", "DB2"): ["KB-0001", "KB-0002", "KB-0003", "KB-0004", "KB-0005"],
    ("Database", "MS SQL Server"): ["KB-0001", "KB-0002", "KB-0003", "KB-0004", "KB-0005"],
    ("Database", "Oracle"): ["KB-0001", "KB-0002", "KB-0003", "KB-0004", "KB-0005"],
    ("Hardware", "Monitor"): ["KB-0008"],
    ("Hardware", "Keyboard"): ["KB-0012"],
    ("Hardware", "Mouse"): ["KB-0012"],
    ("Hardware", "CPU"): [],
    ("Hardware", "Disk"): [],
    ("Hardware", "Memory"): ["KB-0006"],
    ("Software", "Email"): ["KB-0015"],
    ("Software", "Operating System"): ["KB-0006", "KB-0029", "KB-0030"],
    ("Inquiry/Help", "Antivirus"): ["KB-0014", "KB-0026", "KB-0028"],
    ("Inquiry/Help", "Internal Application"): ["KB-0014", "KB-0025"],
}

# === LLM CLIENT HELPER ===
def get_llm_client():
    """Returns an OpenAI-compatible client configured for current provider."""
    from openai import OpenAI
    if not LLM_API_KEY:
        raise RuntimeError("No LLM API key configured — set one in the sidebar.")
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
