
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load the analysis dataset
df = pd.read_csv("results/merged_analysis.csv")

# Keep records with known ratings
df = df.dropna(subset=["rating"]).copy()

# Predict whether a review has a low rating (1 or 2)
df["low_rating"] = (df["rating"] <= 2).astype(int)

# Exclude return_flag because it may not be known
# when we want to predict a rating in advance.
features = [
    "category",
    "brand",
    "price_band",
    "verified",
    "review_length",
    "delivery_days",
    "discount_percent"
]

X = df[features]
y = df["low_rating"]

categorical = [
    "category", "brand", "price_band", "verified"
]
numeric = [
    "review_length", "delivery_days", "discount_percent"
]

preprocessor = ColumnTransformer([
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    ),
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric
    )
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Model 1: Baseline — always predicts the majority class
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

# Model 2: Random Forest
random_forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

random_forest.fit(X_train, y_train)

# Evaluate both models
models = {
    "Baseline": baseline,
    "Random Forest": random_forest
}

for name, model in models.items():
    predictions = model.predict(X_test)

    print(f"\n{'=' * 40}")
    print(name)
    print("=" * 40)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Low-rating Precision:",
          round(precision_score(y_test, predictions, zero_division=0), 3))
    print("Low-rating Recall:",
          round(recall_score(y_test, predictions, zero_division=0), 3))
    print("Low-rating F1-score:",
          round(f1_score(y_test, predictions, zero_division=0), 3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions,
        target_names=["Not Low Rating", "Low Rating"],
        zero_division=0
    ))
