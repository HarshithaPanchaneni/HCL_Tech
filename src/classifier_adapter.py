from src.config import ClassificationResult
from prediction import classify_ticket


def classify_for_pipeline(ticket_text: str) -> ClassificationResult:
    # Get predictions from Person 2's classifier.
    result = classify_ticket(ticket_text)

    # Use the lower confidence as the overall confidence.
    confidence = min(
        result["category_confidence"],
        result["subcategory_confidence"]
    )

    # Convert the classifier dictionary into our shared dataclass.
    return ClassificationResult(
        category=result["category"],
        subcategory=result["subcategory"],
        priority=result["priority"],
        confidence=confidence,
        needs_review=result["needs_review"],
        reasoning=(
            f"Category confidence: {result['category_confidence']}; "
            f"Subcategory confidence: {result['subcategory_confidence']}"
        )
    )