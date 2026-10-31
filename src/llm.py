
import requests  # Allows Python to send HTTP requests to the local Ollama server.
from typing import Any  # Allows us to specify flexible dictionary types for runbook data.

OLLAMA_URL = "http://localhost:11434/api/generate"  # Local Ollama API endpoint.
MODEL_NAME = "phi3.5:latest"  # The Phi 3.5 model we selected.
REQUEST_TIMEOUT = 120  # Maximum time to wait for a response, in seconds.


def generate_resolution(ticket_text: str, runbook: dict[str, Any]) -> str:
    """
    Generate a user-friendly resolution using an approved runbook.

    Args:
        ticket_text: The original IT support ticket.
        runbook: The complete approved or retrieved runbook dictionary.

    Returns:
        A professionally written resolution in English.
    """

    if not runbook:  # Ensure that we have a runbook before asking the LLM for a resolution.
        raise ValueError("Cannot generate a resolution without a runbook.")

    title = runbook.get("title", "IT Support Issue")  # Get the runbook title.
    symptoms = runbook.get("symptoms", [])  # Get the symptoms documented in the runbook.
    diagnosis = runbook.get("diagnosis", [])  # Get the approved diagnostic checks.
    steps = runbook.get("resolution_steps", [])  # Get the approved troubleshooting steps.
    validation = runbook.get("validation", [])  # Get the checks used to confirm resolution.
    escalation = runbook.get("escalate_if", [])  # Get the conditions requiring escalation.

    if not steps:  # Do not generate instructions if the runbook has no resolution steps.
        raise ValueError("The retrieved runbook contains no resolution steps.")

    prompt = f"""
You are an enterprise IT support resolution assistant.

Your task is to explain the supplied runbook to an IT support user.

IMPORTANT RULES:
1. Respond entirely in English.
2. Use only information provided in the ticket and runbook.
3. Do not invent troubleshooting steps, commands, causes, or configurations.
4. Preserve the meaning and order of the approved resolution steps.
5. Clearly distinguish diagnostic checks from actual fixes.
6. Include validation and escalation conditions when relevant.
7. Use simple, professional and concise language.
8. If the runbook does not address the reported issue, say so clearly.

USER TICKET:
{ticket_text}

APPROVED RUNBOOK:
Title: {title}

Symptoms:
{symptoms}

Diagnosis:
{diagnosis}

Approved Resolution Steps:
{steps}

Validation:
{validation}

Escalate If:
{escalation}

Write a clear resolution for the user based strictly on this information.
"""  # Construct the prompt using the ticket and retrieved runbook information.

    payload = {
        "model": MODEL_NAME,  # Tell Ollama which model to use.
        "prompt": prompt,  # Supply the instructions and runbook context.
        "stream": False  # Wait for the complete response instead of receiving streamed chunks.
    }

    try:
        response = requests.post(
            OLLAMA_URL,  # Send the request to our local Ollama server.
            json=payload,  # Send the model name and prompt as JSON.
            timeout=REQUEST_TIMEOUT  # Prevent the request from waiting indefinitely.
        )

        response.raise_for_status()  # Raise an error if the API returns an HTTP error.

        result = response.json()  # Convert Ollama's JSON response into a Python dictionary.
        generated_text = result.get("response", "").strip()  # Extract and clean the generated answer.

        if not generated_text:  # Check that the model actually returned some text.
            raise RuntimeError("Phi 3.5 returned an empty response.")

        return generated_text  # Return the final resolution to the calling backend.

    except requests.exceptions.Timeout as error:  # Handle cases where Phi takes too long.
        raise RuntimeError("Phi 3.5 request timed out.") from error

    except requests.exceptions.RequestException as error:  # Handle connection and HTTP errors.
        raise RuntimeError(f"Could not communicate with Ollama: {error}") from error
