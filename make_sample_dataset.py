"""
make_sample_dataset.py
Generates a synthetic customer-churn CSV so you can test-drive the agent
immediately without needing your own dataset.

Run:
    python sample_data/make_sample_dataset.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000

tenure = rng.integers(0, 72, n)
monthly_charges = np.round(rng.normal(65, 25, n).clip(15, 150), 2)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.20])
tech_support = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])
payment_method = rng.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
)
customer_id = [f"CUST-{i:05d}" for i in range(n)]

# Introduce some missingness
monthly_charges_with_na = monthly_charges.copy()
na_idx = rng.choice(n, size=int(0.05 * n), replace=False)
monthly_charges_with_na[na_idx] = np.nan

# Churn probability driven by a few "real" signals + noise
churn_logit = (
    -0.03 * tenure
    + 0.015 * monthly_charges
    + (contract == "Month-to-month") * 1.2
    + (internet_service == "Fiber optic") * 0.5
    + (tech_support == "No") * 0.6
    - 1.0
)
churn_prob = 1 / (1 + np.exp(-churn_logit))
churn = (rng.random(n) < churn_prob).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df = pd.DataFrame({
    "CustomerID": customer_id,
    "Tenure": tenure,
    "MonthlyCharges": monthly_charges_with_na,
    "Contract": contract,
    "InternetService": internet_service,
    "TechSupport": tech_support,
    "PaymentMethod": payment_method,
    "Churn": churn_label,
})

# Add one constant column and one duplicate row to exercise the cleaner
df["Country"] = "India"
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

out_path = "sample_data/customer_churn.csv"
df.to_csv(out_path, index=False)
print(f"Sample dataset written to {out_path} ({len(df)} rows, {df.shape[1]} columns)")
