import pandas as pd
import os

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv(
    r"D:\Ai_stock_predictor\data\featured\technical_features.csv"
)

# -----------------------------
# Future Price
# -----------------------------

PREDICTION_DAYS = 5

df["Future_Close"] = df["Close"].shift(-PREDICTION_DAYS)

# -----------------------------
# Future Return
# -----------------------------

df["Future_Return"] = (
    (df["Future_Close"] - df["Close"])
    / df["Close"]
)

# -----------------------------
# Generate Labels
# -----------------------------

BUY_THRESHOLD = 0.02     # +2%
SELL_THRESHOLD = -0.02   # -2%

def create_label(future_return):
    if future_return > BUY_THRESHOLD:
        return "Buy"
    elif future_return < SELL_THRESHOLD:
        return "Sell"
    else:
        return "Hold"

df["Label"] = df["Future_Return"].apply(create_label)

# -----------------------------
# Remove last rows with no future price
# -----------------------------

df = df.dropna().reset_index(drop=True)

# -----------------------------
# Display Label Distribution
# -----------------------------

print("\nLabel Distribution:\n")
print(df["Label"].value_counts())

print("\nLabel Percentages:\n")
print(df["Label"].value_counts(normalize=True) * 100)

# -----------------------------
# Save Dataset
# -----------------------------

os.makedirs(
    r"D:\Ai_stock_predictor\data\labeled",
    exist_ok=True
)

df.to_csv(
    r"D:\Ai_stock_predictor\data\labeled\labeled_dataset.csv",
    index=False
)

print("\n✅ Labels generated successfully!")