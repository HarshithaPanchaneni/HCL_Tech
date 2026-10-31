
import joblib
from pathlib import Path

# Find saved models in the project folder
BASE_DIR = Path(__file__).resolve().parent

category_model = joblib.load(BASE_DIR / "category_model.pkl")
subcategory_model = joblib.load(BASE_DIR / "subcategory_model.pkl")

# Subcategory belongs to which category?
SUBCATEGORY_MAP = {
    "Database": ["DB2", "MS SQL Server", "Oracle"],
    "Hardware": ["CPU", "Disk", "Keyboard", "Memory", "Monitor", "Mouse"],
    "Inquiry/Help": ["Antivirus", "Internal Application"],
    "Network": ["DHCP", "DNS", "IP Address", "VPN", "Wireless"],
    "Software": ["Email", "Operating System"]
}

def predict_priority(text):
    text = text.lower()

    if any(word in text for word in [
        "all employees", "entire company", "company-wide outage"
    ]):
        return "Critical"

    if any(word in text for word in [
        "production server down", "major outage",
        "business stopped"
    ]):
        return "High"

    if any(word in text for word in [
        "cannot login", "unable to access",
        "not working", "keeps disconnecting"
    ]):
        return "Medium"

    return "Needs Review"


def classify_ticket(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a valid IT complaint.")

    text = text.strip()

    # Predict main category
    category = category_model.predict([text])[0]

    # Calculate category confidence
    category_probs = category_model.predict_proba([text])[0]
    category_confidence = float(max(category_probs))

    # Get subcategory probabilities
    subcategory_probs = subcategory_model.predict_proba([text])[0]

    # Get all subcategory names
    subcategory_names = subcategory_model.classes_

    # Only consider subcategories belonging to predicted category
    valid_subcategories = SUBCATEGORY_MAP.get(category, [])

    candidates = [
        (name, float(prob))
        for name, prob in zip(subcategory_names, subcategory_probs)
        if name in valid_subcategories
    ]

    if candidates:
        subcategory, subcategory_confidence = max(
            candidates, key=lambda item: item[1]
        )
    else:
        subcategory = "Needs Review"
        subcategory_confidence = 0.0

    # Rule-based priority
    priority = predict_priority(text)

    # Low-confidence handling
    needs_review = (
        category_confidence < 0.70
        or subcategory_confidence < 0.70
    )

    return {
        "category": category,
        "subcategory": subcategory,
        "category_confidence": round(category_confidence, 4),
        "subcategory_confidence": round(subcategory_confidence, 4),
        "priority": priority,
        "needs_review": needs_review
    }


# Test without Streamlit
if __name__ == "__main__":
    complaint = input("Enter IT complaint: ")
    result = classify_ticket(complaint)

    print("\nCLASSIFICATION RESULT")
    for key, value in result.items():
        print(f"{key}: {value}")
