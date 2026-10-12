
import pandas as pd
from pathlib import Path

data_dir = Path("data")

files = [
    "reviews.csv",
    "products.csv",
    "sellers.csv",
    "orders.csv",
    "support.csv"
]

for filename in files:
    print("\n" + "=" * 45)
    print("DATASET:", filename)
    print("=" * 45)

    df = pd.read_csv(data_dir / filename)

    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())
