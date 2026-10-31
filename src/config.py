
import os  # Reads optional configuration from environment variables.
from pathlib import Path  # Builds file paths safely across operating systems.
from dataclasses import dataclass, field  # Defines shared structured data models.
from typing import Optional  # Allows fields to contain None.

# -------------------- PROJECT PATHS --------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Finds the repository root.
VECTOR_DB_PATH = "data/processed/chroma_db"  # Matches the database path used by both indexers.
KB_PATH = "data/knowledge_base/articles.json"  # Location of the KB article JSON file.
KB_MAPPING_PATH = "data/knowledge_base/subcategory_to_kb.json"  # Location of the category-to-KB mapping.

# -------------------- DATABASE CONFIGURATION --------------------

TICKET_COLLECTION = "tickets"  # Historical ticket collection name.
KB_COLLECTION = "knowledge_base"  # Approved knowledge-base collection name.

# -------------------- EMBEDDING CONFIGURATION --------------------

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Must match the model used by both indexing pipelines.

# -------------------- LLM CONFIGURATION --------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()  # Uses Ollama by default; can be changed through an environment variable.

if LLM_PROVIDER == "ollama":  # Configure the locally running Ollama server.
    LLM_MODEL = "phi3.5:latest"  # Uses the Phi 3.5 model already installed on your laptop.
    LLM_BASE_URL = "http://localhost:11434/v1"  # OpenAI-compatible Ollama endpoint.
    LLM_API_KEY = "ollama"  # Ollama does not require a real API key.
elif LLM_PROVIDER == "groq":  # Keep Groq available as an optional alternative.
    LLM_MODEL = "llama-3.1-70b-versatile"  # Groq model used by your teammate's original configuration.
    LLM_BASE_URL = "https://api.groq.com/openai/v1"  # Groq OpenAI-compatible endpoint.
    LLM_API_KEY = os.getenv("GROQ_API_KEY")  # Reads the Groq API key from the environment.
else:  # Reject unsupported providers rather than silently using the wrong configuration.
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")

LLM_TIMEOUT = 120  # Maximum time allowed for an LLM response, in seconds.

# -------------------- RETRIEVAL CONFIGURATION --------------------

TOP_K_SIMILAR = 5  # Maximum number of historical tickets to retrieve.
TOP_K_KB = 3  # Maximum number of KB articles to retrieve.
KNN_MIN_SIMILARITY = 0.75  # Initial historical-ticket similarity threshold; calibrate with evaluation.
KB_MIN_SCORE = 0.75  # Initial KB similarity threshold; calibrate with evaluation.

# -------------------- CLASSIFICATION CONFIGURATION --------------------

CONFIDENCE_THRESHOLD = 0.7  # Initial threshold for deciding whether classification needs review.
PRIORITY_ESCALATE = ["High", "P1"]  # Priorities that should trigger escalation consideration.

# -------------------- SHARED DATA MODELS --------------------
# These models are shared across the team; do not change their field contracts without coordination.

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
    priority: str
    confidence: float
    needs_review: bool
    reasoning: str
    secondary_category: Optional[str] = None
    secondary_subcategory: Optional[str] = None
    secondary_confidence: Optional[float] = None

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
    retrieval_source: str = "vector"

@dataclass
class Evidence:
    source_id: str
    source_type: str
    relevant_excerpt: str
    support_reason: str

@dataclass
class ResolutionRecommendation:
    recommended_resolution: str
    confidence: float
    evidence: list
    alternative_resolutions: list
    needs_escalation: bool
    knowledge_articles_used: list

@dataclass
class FeedbackRecord:
    ticket_id: str
    feedback_type: str
    classification_correct: Optional[bool] = None
    correct_category: Optional[str] = None
    correct_subcategory: Optional[str] = None
    retrieval_relevant: Optional[bool] = None
    resolution_helpful: Optional[bool] = None
    analyst_notes: Optional[str] = None
    missing_kb_topic: Optional[str] = None

@dataclass
class AnalysisResult:
    """Final output combining all stages."""
    input_ticket: Ticket
    classification: ClassificationResult
    similar_tickets: list
    knowledge_articles: list
    resolution: ResolutionRecommendation

# -------------------- DATASET TAXONOMY --------------------

TAXONOMY = {
    "Database": ["DB2", "MS SQL Server", "Oracle"],
    "Hardware": ["CPU", "Disk", "Keyboard", "Memory", "Monitor", "Mouse"],
    "Network": ["DHCP", "DNS", "IP Address", "VPN", "Wireless"],
    "Software": ["Email", "Operating System"],
    "Inquiry/Help": ["Antivirus", "Internal Application"],
}

# -------------------- SUBCATEGORY TO KB MAPPING --------------------

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
