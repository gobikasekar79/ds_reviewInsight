
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# 1. Load dataset
df = pd.read_csv("results/merged_analysis.csv")
df = df.dropna(subset=["rating"]).copy()

# 2. Create target: 1 = low rating, 0 = not low
df["low_rating"] = (df["rating"] <= 2).astype(int)

# 3. Select features
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

categorical = ["category", "brand", "price_band", "verified"]
numeric = ["review_length", "delivery_days", "discount_percent"]

# 4. Preprocessing
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

# 5. Split data into training, validation and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.625,
    random_state=42,
    stratify=y_temp
)

# 6. Train model using training data only
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

# 7. Select threshold using validation data
val_probabilities = model.predict_proba(X_val)[:, 1]

print("\nVALIDATION RESULTS")
print("-" * 55)

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

for threshold in thresholds:
    predictions = (val_probabilities >= threshold).astype(int)

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision_score(y_val, predictions, zero_division=0):.3f} | "
        f"Recall: {recall_score(y_val, predictions, zero_division=0):.3f} | "
        f"F1: {f1_score(y_val, predictions, zero_division=0):.3f}"
    )

# Choose the threshold with the highest validation F1-score
best_threshold = max(
    thresholds,
    key=lambda t: f1_score(
        y_val,
        (val_probabilities >= t).astype(int),
        zero_division=0
    )
)

print(f"\nSelected threshold: {best_threshold:.2f}")

# 8. Evaluate the selected threshold on untouched test data
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

print("\nFINAL TEST RESULTS")
print("-" * 55)
print("Accuracy:", round(accuracy_score(y_test, test_predictions), 3))
print("Precision:", round(precision_score(y_test, test_predictions, zero_division=0), 3))
print("Recall:", round(recall_score(y_test, test_predictions, zero_division=0), 3))
print("F1-score:", round(f1_score(y_test, test_predictions, zero_division=0), 3))

from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, test_predictions)

print("\nCONFUSION MATRIX")
print("[[True Not-Low, Predicted Low],")
print(" [Actual Low, Predicted Not-Low]]")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nDETAILS")
print("Correctly identified not-low reviews:", tn)
print("False positives (flagged incorrectly):", fp)
print("Missed low-rated reviews:", fn)
print("Correctly identified low-rated reviews:", tp)

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["Not Low Rating", "Low Rating"],
    zero_division=0
))
