
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load the combined dataset created during EDA
DATA_PATH = Path("results/merged_analysis.csv")
OUTPUT_DIR = Path("results")
df = pd.read_csv(DATA_PATH)

# 2. Prepare the target
# Low rating = 1 or 2; not low = 3, 4 or 5
df = df.dropna(subset=["rating"]).copy()
df["low_rating"] = (df["rating"] <= 2).astype(int)

# 3. Select features available in our merged dataset
# Exclude rating itself and identifiers to prevent direct leakage.
feature_cols = [
    "category",
    "brand",
    "price_band",
    "verified",
    "review_length",
    "delivery_days",
    "discount_percent",
    "return_flag"
]

X = df[feature_cols].copy()
y = df["low_rating"]

categorical_cols = [
    "category", "brand", "price_band", "verified", "return_flag"
]
numeric_cols = [
    "review_length", "delivery_days", "discount_percent"
]

# 4. Preprocess features inside the ML pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_cols
        ),
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric_cols
        )
    ]
)

# 5. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 6. Build and train the model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# 7. Predict on unseen test data
y_pred = model.predict(X_test)

# 8. Evaluate performance
print("MODEL: Random Forest Classifier")
print("Target: Low rating (1-2) vs not low (3-5)")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Low-rating records:", int(y.sum()))
print("Not-low-rating records:", int((y == 0).sum()))

print("\nMODEL PERFORMANCE")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("Precision:", round(
    precision_score(y_test, y_pred, zero_division=0), 3
))
print("Recall:", round(
    recall_score(y_test, y_pred, zero_division=0), 3
))
print("F1-score:", round(
    f1_score(y_test, y_pred, zero_division=0), 3
))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test, y_pred,
    target_names=["Not low rating", "Low rating"],
    zero_division=0
))

print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

# 9. Save predictions for inspection
test_results = X_test.copy()
test_results["actual_low_rating"] = y_test
test_results["predicted_low_rating"] = y_pred
test_results.to_csv(
    OUTPUT_DIR / "ml_test_predictions.csv", index=False
)

print("\nSaved test predictions to results/ml_test_predictions.csv")
