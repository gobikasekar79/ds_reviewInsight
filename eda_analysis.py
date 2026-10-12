
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path("data/cleaned")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load cleaned datasets
reviews = pd.read_csv(DATA_DIR / "reviews_clean.csv")
products = pd.read_csv(DATA_DIR / "products_clean.csv")
sellers = pd.read_csv(DATA_DIR / "sellers_clean.csv")
orders = pd.read_csv(DATA_DIR / "orders_clean.csv")
support = pd.read_csv(DATA_DIR / "support_clean.csv")

# 2. Merge reviews with product and seller information
df = reviews.merge(
    products, on="product_id", how="left",
    validate="many_to_one"
)

df = df.merge(
    sellers, on="seller_id", how="left",
    validate="many_to_one"
)

# Merge order details without duplicating review rows
df = df.merge(
    orders[[
        "review_id", "delivery_days",
        "discount_percent", "return_flag"
    ]],
    on="review_id", how="left",
    validate="one_to_one"
)

# Save the combined analysis dataset
df.to_csv(OUTPUT_DIR / "merged_analysis.csv", index=False)

print("Combined dataset shape:", df.shape)
print("\nMissing values in combined dataset:")
print(df.isnull().sum())

# Ratings are excluded only from calculations requiring a rating
rated = df.dropna(subset=["rating"]).copy()

print("\nValid ratings available for analysis:", len(rated))
print("Overall average rating:", round(rated["rating"].mean(), 2))

# 3. Average rating by product category
category_summary = (
    rated.groupby("category")["rating"]
    .agg(["mean", "count"])
    .round(2)
    .sort_values("mean")
)

print("\nAVERAGE RATING BY CATEGORY")
print(category_summary)

category_summary.to_csv(OUTPUT_DIR / "category_ratings.csv")

category_summary["mean"].plot(
    kind="bar", title="Average Rating by Product Category",
    ylabel="Average Rating (1-5)", xlabel="Category",
    ylim=(0, 5), rot=30
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_ratings.png")
plt.show()

# 4. Verified vs unverified reviews
verified_summary = (
    rated.groupby("verified")["rating"]
    .agg(["mean", "count"])
    .round(2)
)

print("\nVERIFIED VS UNVERIFIED REVIEWS")
print(verified_summary)

verified_summary.to_csv(OUTPUT_DIR / "verified_comparison.csv")

# 5. Delivery time vs rating
delivery_summary = (
    rated.groupby("delivery_days")["rating"]
    .agg(["mean", "count"])
    .round(2)
    .sort_index()
)

print("\nDELIVERY DAYS VS RATING")
print(delivery_summary)

print(
    "Delivery-rating correlation:",
    round(rated["delivery_days"].corr(rated["rating"]), 3)
)

delivery_summary.to_csv(OUTPUT_DIR / "delivery_ratings.csv")

delivery_summary["mean"].plot(
    marker="o", title="Delivery Days vs Average Rating",
    xlabel="Delivery Days", ylabel="Average Rating (1-5)",
    ylim=(0, 5)
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "delivery_ratings.png")
plt.show()

# 6. Return status vs rating
return_summary = (
    rated.groupby("return_flag")["rating"]
    .agg(["mean", "count"])
    .round(2)
)

print("\nRETURN STATUS VS RATING")
print(return_summary)

return_summary.to_csv(OUTPUT_DIR / "return_ratings.csv")

# 7. Seller comparison
seller_summary = (
    rated.groupby(["seller_name", "region"])["rating"]
    .agg(["mean", "count"])
    .round(2)
    .sort_values("mean")
)

print("\nSELLER RATINGS (at least 3 rated reviews)")
print(seller_summary[seller_summary["count"] >= 3])

seller_summary.to_csv(OUTPUT_DIR / "seller_ratings.csv")

# 8. Complaint summary, joined using review_id
support_analysis = support.merge(
    df[["review_id", "rating"]],
    on="review_id",
    how="left",
    validate="many_to_one"
)

complaint_summary = (
    support_analysis.groupby("complaint_category")
    .agg(
        tickets=("ticket_id", "count"),
        average_rating=("rating", "mean")
    )
    .round(2)
    .sort_values("tickets", ascending=False)
)

print("\nCOMPLAINT SUMMARY")
print(complaint_summary)

complaint_summary.to_csv(OUTPUT_DIR / "complaint_summary.csv")

print("\nEDA completed! Results saved in the results folder.")
