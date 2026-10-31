
import requests  # Sends HTTP requests to the local Ollama server.
from typing import Any  # Allows the function to accept dictionaries or dataclass objects.
import json  # Formats retrieved KB information clearly inside the prompt.

OLLAMA_URL = "http://localhost:11434/api/generate"  # Local Ollama generation endpoint.
MODEL_NAME = "phi3.5:latest"  # Phi 3.5 model already installed on your laptop.
REQUEST_TIMEOUT = 120  # Maximum time to wait for model generation, in seconds.


def _get_field(article: Any, field: str, default: Any = None) -> Any:
    # Read a field from either a dictionary or a Python dataclass object.
    if isinstance(article, dict):
        return article.get(field, default)

    # Use getattr when the retrieved article is a KnowledgeArticle object.
    return getattr(article, field, default)


def generate_resolution(ticket_text: str, knowledge_articles: list[Any]) -> str:
    """
    Generate an English IT support resolution using retrieved KB articles.

    Args:
        ticket_text: The original IT support ticket description.
        knowledge_articles: Retrieved KnowledgeArticle objects or dictionaries.

    Returns:
        A user-friendly resolution grounded in the supplied KB articles.
    """

    # Refuse to generate a resolution without approved KB information.
    if not knowledge_articles:
        raise ValueError("Cannot generate a resolution without KB articles.")

    # Store formatted information from every retrieved KB article.
    article_context = []

    # Process each retrieved article individually.
    for article in knowledge_articles:

        # Extract the article's identifier.
        kb_id = _get_field(article, "kb_id", "Unknown KB ID")

        # Extract the article title.
        title = _get_field(article, "title", "Untitled article")

        # Extract the documented symptoms.
        symptoms = _get_field(article, "symptoms", [])

        # Extract the approved diagnostic steps.
        diagnostic_steps = _get_field(article, "diagnostic_steps", [])

        # Extract the approved resolution steps.
        # Resolution steps are nested inside the KB article's "resolution" object.
        # KnowledgeArticle is a dataclass, so read its resolution_steps attribute.
        resolution_steps = _get_field(article, "resolution_steps", [])

        # Extract the documented escalation conditions.
        escalation_criteria = _get_field(article, "escalation_criteria", "")

        # Skip articles that do not contain actual resolution instructions.
        if not resolution_steps:
            continue

        # Format the article's information for the LLM prompt.
        article_context.append({
            "kb_id": kb_id,
            "title": title,
            "symptoms": symptoms,
            "diagnostic_steps": diagnostic_steps,
            "resolution_steps": resolution_steps,
            "escalation_criteria": escalation_criteria
        })

    # Stop if none of the retrieved articles contains usable resolution steps.
    if not article_context:
        raise ValueError("Retrieved KB articles contain no resolution steps.")

    # Convert the KB information into readable JSON for the model.
    formatted_articles = json.dumps(article_context, indent=2, ensure_ascii=False)

    # Construct a prompt that restricts the model to approved KB information.
    prompt = f"""
You are an enterprise IT support resolution assistant.

Your task is to explain the supplied approved knowledge-base instructions
clearly to an IT support user.

STRICT RULES:
1. Respond entirely in English.
2. Use only information explicitly provided in the ticket and KB articles.
3. Never invent commands, troubleshooting steps, causes, or configurations.
4. Do not add recommendations based on general IT knowledge.
5. Preserve the original order and meaning of all supplied steps.
6. Keep diagnostic checks separate from resolution steps.
7. Do not present diagnostic checks as approved fixes.
8. Do not claim that the issue has been resolved.
9. If the KB does not adequately address the ticket, state this clearly.
10. Keep the response concise and professional.
11. Do not treat historical ticket descriptions as verified solutions.

USER TICKET:
{ticket_text}

RETRIEVED APPROVED KNOWLEDGE-BASE ARTICLES:
{formatted_articles}

Generate a concise response using exactly these sections:

1. ISSUE: One sentence describing the ticket.

2. DIAGNOSTICS: List only the supplied diagnostic_steps.
Maximum 3 bullet points.

3. RESOLUTION: List only the supplied resolution_steps.
Preserve their original wording and order. Do not add explanations.

4. ESCALATION: State the supplied escalation_criteria in one or two sentences.

STRICT:
- Use only the KB article most relevant to the ticket.
- Never combine steps from unrelated articles.
- Never infer or invent information.
- Do not summarize, merge, omit, or reinterpret technical steps.
- Include every supplied diagnostic and resolution step exactly once.
- Copy escalation criteria faithfully without adding inferred conditions.
- Maximum 200 words.
"""

    # Prepare the request payload for Ollama.
    payload = {
    "model": MODEL_NAME,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0,
        "num_predict": 300
    }
}

    try:
        # Send the generation request to the local Ollama server.
        response = requests.post(
            OLLAMA_URL,  # Target the local Ollama API.
            json=payload,  # Send the model and prompt as JSON.
            timeout=REQUEST_TIMEOUT  # Prevent indefinite waiting.
        )

        # Raise an exception if Ollama returns an HTTP error.
        response.raise_for_status()

        # Parse Ollama's JSON response.
        result = response.json()

        # Extract and clean the generated resolution.
        generated_text = result.get("response", "").strip()

        # Reject empty model responses.
        if not generated_text:
            raise RuntimeError("Phi 3.5 returned an empty response.")

        # Return the generated resolution to the calling pipeline.
        return generated_text

    except requests.exceptions.Timeout as error:
        # Handle requests that exceed the configured timeout.
        raise RuntimeError("Phi 3.5 request timed out.") from error

    except requests.exceptions.RequestException as error:
        # Handle connection failures and HTTP errors.
        raise RuntimeError(f"Could not communicate with Ollama: {error}") from error
