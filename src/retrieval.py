
from pathlib import Path  # Helps construct file paths independently of the operating system.
from functools import lru_cache  # Caches objects so we don't repeatedly load expensive resources.

import chromadb  # Provides the vector database client.
from sentence_transformers import SentenceTransformer  # Loads our E5 embedding model.

# Resolve the project root directory (the parent of the src folder).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Store the ChromaDB database inside the project root.
CHROMA_PATH = PROJECT_ROOT / "chroma_db"

# Name of the multilingual embedding model agreed upon by the team.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"


@lru_cache(maxsize=1)
def get_chroma_client():
    # Create a persistent ChromaDB client and reuse it across function calls.
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


@lru_cache(maxsize=1)
def get_embedding_model():
    # Load the E5 model only when first needed, then reuse it.
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_query(text: str) -> list[float]:
    # Add the required E5 prefix to identify this text as a search query.
    prefixed_text = f"query: {text}"

    # Generate a normalized embedding so cosine similarity works consistently.
    embedding = get_embedding_model().encode(
        prefixed_text,
        normalize_embeddings=True
    )

    # Convert the NumPy embedding into a regular Python list for ChromaDB.
    return embedding.tolist()
