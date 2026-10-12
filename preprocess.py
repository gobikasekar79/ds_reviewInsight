
import pandas as pd
from pathlib import Path

# Folder paths
DATA_DIR = Path("data")
CLEAN_DIR = DATA_DIR / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
reviews = pd.read_csv(DATA_DIR / "reviews.csv")
products = pd.read_csv(DATA_DIR / "products.csv")
sellers = pd.read_csv(DATA_DIR / "sellers.csv")
orders = pd.read_csv(DATA_DIR / "orders.csv")
support = pd.read_csv(DATA_DIR / "support.csv")

print("STEP 1: REMOVE DUPLICATE REVIEWS")

before = len(reviews)
reviews = reviews.drop_duplicates().copy()
print("Duplicate rows removed:", before - len(reviews))
print("Reviews remaining:", len(reviews))

print("\nSTEP 2: CHECK MISSING RATINGS")

print("Missing ratings before handling:", reviews["rating"].isna().sum())

# Keep missing ratings in the cleaned review file.
# We will exclude them only when an analysis requires a known rating.
print("Missing ratings retained for now; no ratings invented.")

print("\nSTEP 3: STANDARDIZE CATEGORIES")

print("Categories before:", products["category"].unique())

products["category"] = (
    products["category"]
    .astype("string")
    .str.strip()
    .str.title()
)

print("Categories after:", products["category"].unique())

print("\nSTEP 4: CHECK EXTREME REVIEW LENGTHS")

q1 = reviews["review_length"].quantile(0.25)
q3 = reviews["review_length"].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr

reviews["long_review_flag"] = (
    reviews["review_length"] > upper_limit
)

print("IQR upper limit:", round(upper_limit, 2))
print("Potential extreme reviews:", reviews["long_review_flag"].sum())
print("Extreme values flagged, not automatically deleted.")

print("\nSTEP 5: CHECK UNMATCHED PRODUCT IDs IN ORDERS")

valid_products = set(products["product_id"])
valid_reviews = reviews[["review_id", "product_id"]].drop_duplicates(
    subset="review_id"
)

orders = orders.merge(
    valid_reviews,
    on="review_id",
    how="left",
    suffixes=("_order", "_review"),
    validate="many_to_one"
)

mismatch = (
    orders["product_id_order"].notna()
    & orders["product_id_order"].ne(orders["product_id_review"])
)

print("Order product IDs needing correction:", int(mismatch.sum()))

# The review ID provides the product link for these sample records.
orders["product_id"] = orders["product_id_review"]

# Check whether any order still refers to an unknown product.
unmatched = ~orders["product_id"].isin(valid_products)
print("Orders still unmatched to products:", int(unmatched.sum()))

orders = orders.drop(
    columns=["product_id_order", "product_id_review"]
)

print("\nSTEP 6: SAVE CLEANED DATASETS")

datasets = {
    "reviews_clean.csv": reviews,
    "products_clean.csv": products,
    "sellers_clean.csv": sellers,
    "orders_clean.csv": orders,
    "support_clean.csv": support
}

for filename, df in datasets.items():
    df.to_csv(CLEAN_DIR / filename, index=False)
    print(f"{filename}: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nPreprocessing completed!")
print("Cleaned files saved in:", CLEAN_DIR)
