
from prediction import classify_ticket

tickets = [
    "My laptop keyboard is not working",
    "I cannot connect to the office VPN",
    "Oracle database is showing an error",
    "My Outlook email is not opening",
    "All employees are unable to connect to WiFi"
]

for ticket in tickets:
    result = classify_ticket(ticket)

    print("\nTicket:", ticket)
    print("Category:", result["category"])
    print("Subcategory:", result["subcategory"])
    print("Category Confidence:",
          round(result["category_confidence"] * 100, 2), "%")
    print("Subcategory Confidence:",
          round(result["subcategory_confidence"] * 100, 2), "%")
    print("Priority:", result["priority"])
    print("Human Review:", result["needs_review"])
    print("-" * 40)
