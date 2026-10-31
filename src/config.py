
from pathlib import Path  # Provides reliable cross-platform path handling.

# Find the project root directory (the parent of the src folder).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -------------------- DATABASE CONFIGURATION --------------------

CHROMA_PATH = PROJECT_ROOT / "chroma_db"  # Location of the persistent ChromaDB database.

TICKETS_COLLECTION = "tickets"  # Collection containing historical IT support tickets.
KB_COLLECTION = "knowledge_base"  # Collection containing complete troubleshooting articles.

# -------------------- EMBEDDING CONFIGURATION --------------------

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"  # Shared embedding model agreed upon by the team.

# E5 requires different prefixes for stored documents and search queries.
QUERY_PREFIX = "query: "  # Prefix applied when embedding incoming search queries.
PASSAGE_PREFIX = "passage: "  # Prefix used when embedding stored tickets and KB representations.

# -------------------- LLM CONFIGURATION --------------------

OLLAMA_URL = "http://localhost:11434/api/generate"  # Local Ollama text-generation endpoint.
LLM_MODEL = "phi3.5:latest"  # Local model selected for resolution generation.
LLM_TIMEOUT = 120  # Maximum time to wait for the LLM response, in seconds.

# -------------------- RETRIEVAL CONFIGURATION --------------------

TICKET_TOP_K = 5  # Maximum number of similar historical tickets to retrieve.
KB_TOP_K = 3  # Maximum number of relevant knowledge-base articles to retrieve.

# Initial thresholds only; these must be calibrated using the held-out test set.
KNN_MIN_SIMILARITY = 0.75  # Initial minimum similarity for historical ticket matches.
KB_MIN_SCORE = 0.75  # Initial minimum similarity for knowledge-base matches.

# -------------------- CLASSIFICATION CONFIGURATION --------------------

CLASSIFIER_MIN_CONF = 0.80  # Initial minimum classifier confidence threshold.
PRIORITY_ESCALATE = ["High"]  # Priorities that should trigger escalation consideration.
