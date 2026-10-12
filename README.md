# Challenge 26 — Product Review Insight

This project uses **synthetic sample data** created for practice. It is not real customer data.

## Files
- `data/reviews.csv` — review ratings and metadata (200 rows, including controlled duplicate records)
- `data/products.csv` — product category, brand, price band, and seller
- `data/sellers.csv` — seller region and tenure
- `data/orders.csv` — delivery days, discount, return flag, linked to reviews
- `data/support.csv` — complaint categories and resolutions

## Run
From this folder:
```bash
python generate_datasets.py
```

## Intentional data-quality issues
- Missing ratings
- Duplicate review records
- Inconsistent category labels
- Extreme review lengths
- Unmatched product IDs in orders

Do not report any insight as a confirmed result until you calculate it from the data.
