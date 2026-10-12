"""
Challenge 26 - Product Review Insight
Creates five linked SYNTHETIC sample CSV datasets for learning and analysis.
Run from the project folder with: python generate_datasets.py
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd

SEED = 26
random.seed(SEED)
np.random.seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1) Sellers: 15 synthetic sellers
regions = ["North", "South", "East", "West", "Central"]
seller_rows = []
for i in range(1, 16):
    seller_rows.append({
        "seller_id": f"S{i:03d}",
        "seller_name": f"Seller_{i:02d}",
        "region": random.choice(regions),
        "tenure_years": random.randint(1, 12)
    })
sellers = pd.DataFrame(seller_rows)

# 2) Products: 60 synthetic products linked to sellers
category_brands = {
    "Electronics": ["NovaTech", "ByteWorks", "Orbit"],
    "Home": ["NestWell", "CasaCo", "DailyLiving"],
    "Beauty": ["GlowLab", "PureBloom", "DermaLeaf"],
    "Sports": ["ActivePeak", "MoveMore", "TrailPro"],
    "Books": ["PageTurner", "InkHouse", "Readify"],
}
price_bands = ["Budget", "Mid-range", "Premium"]
products_rows = []
for i in range(1, 61):
    category = list(category_brands.keys())[(i - 1) % len(category_brands)]
    brand = random.choice(category_brands[category])
    products_rows.append({
        "product_id": f"P{i:03d}",
        "category": category,
        "brand": brand,
        "price_band": random.choices(price_bands, weights=[4, 4, 2], k=1)[0],
        "seller_id": f"S{random.randint(1, 15):03d}"
    })
products = pd.DataFrame(products_rows)

# Controlled category inconsistency (to be standardized in preprocessing)
products.loc[products["category"] == "Electronics", "category"] = products.loc[
    products["category"] == "Electronics", "category"
].replace({"Electronics": "electronics"})
# Keep some canonical categories too, so cleaning has a visible effect.
products.loc[products.index[0], "category"] = "Electronics"

# 3) Reviews: exactly 200 rows, including 5 duplicate review records and 6 missing ratings.
# Ratings are deliberately associated with some plausible patterns for analysis practice.
review_rows = []
product_ids = products["product_id"].tolist()
for i in range(1, 196):  # 195 unique source reviews
    product_id = random.choice(product_ids)
    prod = products.loc[products["product_id"] == product_id].iloc[0]
    category = str(prod["category"]).lower()
    # Baseline rating with designed patterns: Electronics/Home slightly lower on average.
    rating_weights = [0.10, 0.16, 0.22, 0.29, 0.23]
    if category in ("electronics", "home"):
        rating_weights = [0.16, 0.22, 0.25, 0.23, 0.14]
    rating = random.choices([1, 2, 3, 4, 5], weights=rating_weights, k=1)[0]
    delivery_days = random.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[5, 15, 22, 20, 15, 10, 7, 6], k=1)[0]
    # Delays are associated with lower ratings to make the relationship analyzable.
    if delivery_days >= 6 and random.random() < 0.60:
        rating = random.choice([1, 2, 3])
    if delivery_days <= 3 and random.random() < 0.55:
        rating = random.choice([4, 5])
    verified = random.choice([True, False])
    review_length = max(5, int(np.random.lognormal(mean=3.7, sigma=0.75)))
    # Controlled extreme review lengths
    if i in (18, 76, 131):
        review_length = random.choice([1800, 2400, 3200])
    review_date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=random.randint(0, 364))
    review_rows.append({
        "review_id": f"R{i:04d}",
        "product_id": product_id,
        "rating": rating,
        "review_length": review_length,
        "verified": verified,
        "review_date": review_date.strftime("%Y-%m-%d")
    })

reviews = pd.DataFrame(review_rows)

# Missing ratings: six values are intentionally set to NaN.
missing_indices = random.sample(list(reviews.index), 6)
reviews.loc[missing_indices, "rating"] = np.nan

# Add five duplicate rows, so reviews.csv contains exactly 200 rows.
duplicate_rows = reviews.sample(n=5, random_state=SEED)
reviews = pd.concat([reviews, duplicate_rows], ignore_index=True)
reviews = reviews.sample(frac=1, random_state=SEED).reset_index(drop=True)

# 4) Orders: one row per original unique review ID, linked through review_id.
order_rows = []
for _, rev in reviews.drop_duplicates(subset=["review_id"], keep="first").iterrows():
    review_id = rev["review_id"]
    product_id = rev["product_id"]
    # Use a designed delay distribution and return probability tied to lower ratings.
    delivery_days = random.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[5, 15, 22, 20, 15, 10, 7, 6], k=1)[0]
    rating = rev["rating"]
    low_rating = pd.notna(rating) and rating <= 2
    if delivery_days >= 6 and random.random() < 0.55:
        delivery_days = random.randint(6, 9)
    return_flag = random.random() < (0.42 if low_rating else 0.12)
    order_rows.append({
        "order_id": f"O{len(order_rows)+1:04d}",
        "review_id": review_id,
        "product_id": product_id,
        "delivery_days": delivery_days,
        "discount_percent": random.choice([0, 5, 10, 15, 20, 25, 30]),
        "return_flag": bool(return_flag)
    })
orders = pd.DataFrame(order_rows)

# Controlled unmatched product IDs in three order rows (data-quality issue to detect).
if len(orders) >= 3:
    orders.loc[orders.index[:3], "product_id"] = ["P999", "P998", "P997"]

# 5) Support: 70 synthetic support records linked to reviews/products.
support_rows = []
review_pool = reviews.drop_duplicates(subset=["review_id"], keep="first")
chosen = review_pool.sample(n=70, random_state=SEED)
complaints = ["Late delivery", "Damaged item", "Wrong item", "Product quality", "Refund issue"]
resolutions = ["Resolved", "Pending", "Escalated"]
for j, (_, rev) in enumerate(chosen.iterrows(), start=1):
    support_rows.append({
        "ticket_id": f"T{j:04d}",
        "review_id": rev["review_id"],
        "product_id": rev["product_id"],
        "complaint_category": random.choice(complaints),
        "resolution": random.choices(resolutions, weights=[7, 2, 1], k=1)[0]
    })
support = pd.DataFrame(support_rows)

# Save files
datasets = {
    "reviews.csv": reviews,
    "products.csv": products,
    "sellers.csv": sellers,
    "orders.csv": orders,
    "support.csv": support,
}
for filename, frame in datasets.items():
    frame.to_csv(DATA_DIR / filename, index=False)

print("Created five synthetic CSV files in:", DATA_DIR)
for filename, frame in datasets.items():
    print(f"{filename}: {len(frame)} rows, {len(frame.columns)} columns")
print("\nNotes: the data intentionally includes missing ratings, duplicate review rows,")
print("inconsistent category labels, extreme review lengths, and unmatched product IDs.")
print("These are synthetic data for practice, not real customer records.")
