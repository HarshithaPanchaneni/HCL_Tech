
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_excel("Dataset_ticket.xlsx")

# Remove missing and duplicate records
df = df.dropna(subset=["Text", "Category", "Subcategory"])
df = df.drop_duplicates(subset=["Text", "Category", "Subcategory"])

# Keep text and labels
X = df["Text"].astype(str)
y_category = df["Category"].astype(str)
y_subcategory = df["Subcategory"].astype(str)

# 2. Split data (same split for both targets)
X_train, X_test, cat_train, cat_test, sub_train, sub_test = train_test_split(
    X, y_category, y_subcategory,
    test_size=0.2,
    random_state=42,
    stratify=y_subcategory
)

# 3. Category classifier
category_model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000)
)

category_model.fit(X_train, cat_train)

# 4. Subcategory classifier
subcategory_model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000)
)

subcategory_model.fit(X_train, sub_train)

# 5. Evaluate
cat_predictions = category_model.predict(X_test)
sub_predictions = subcategory_model.predict(X_test)

print("Category Accuracy:",
      accuracy_score(cat_test, cat_predictions))

print("Subcategory Accuracy:",
      accuracy_score(sub_test, sub_predictions))

print("\nCATEGORY REPORT:")
print(classification_report(cat_test, cat_predictions, zero_division=0))

# 6. Save trained models
joblib.dump(category_model, "category_model.pkl")
joblib.dump(subcategory_model, "subcategory_model.pkl")

print("\nModels saved successfully!")

# 7. Test a new complaint
complaint = ["My internet connection keeps disconnecting"]

category = category_model.predict(complaint)[0]
subcategory = subcategory_model.predict(complaint)[0]

confidence = category_model.predict_proba(complaint).max() * 100

print("\nNEW TICKET:", complaint[0])
print("Category:", category)
print("Subcategory:", subcategory)
print("Category Confidence:", round(confidence, 2), "%")

from sklearn.metrics import f1_score, confusion_matrix

# Predict categories on unseen test data
category_predictions = category_model.predict(X_test)

# Predict subcategories on unseen test data
subcategory_predictions = subcategory_model.predict(X_test)

# Calculate F1 scores
category_f1 = f1_score(
    cat_test,
    category_predictions,
    average="weighted"
)

subcategory_f1 = f1_score(
    sub_test,
    subcategory_predictions,
    average="weighted"
)

print("\n===== FINAL MODEL EVALUATION =====")

print("Category Accuracy:",
      round(accuracy_score(cat_test, category_predictions) * 100, 2), "%")

print("Subcategory Accuracy:",
      round(accuracy_score(sub_test, subcategory_predictions) * 100, 2), "%")

print("Category F1 Score:", round(category_f1, 3))

print("Subcategory F1 Score:", round(subcategory_f1, 3))
print("\nCategory distribution:")
print(df["Category"].value_counts())

print("\nSubcategory distribution:")
print(df["Subcategory"].value_counts())
