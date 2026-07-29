import pandas as pd
import os

# ---------- Load Dataset ----------

master = pd.read_csv(
    r"D:\Ai_stock_predictor\data\processed\master_dataset.csv"
)

# ---------- Convert Date ----------

master["Date"] = pd.to_datetime(master["Date"])

# ---------- Sort by Date ----------

master = master.sort_values(by="Date")

# ---------- Remove Duplicate Rows ----------

master = master.drop_duplicates()

# ---------- Check Missing Values ----------

print("\nMissing Values Before Cleaning:\n")
print(master.isnull().sum())

# ---------- Fill Missing Values ----------

# Forward fill uses the previous day's value
master = master.ffill()

# Backward fill any remaining missing values
master = master.bfill()

# ---------- Remove Any Remaining Missing Rows ----------

master = master.dropna()

# ---------- Reset Index ----------

master = master.reset_index(drop=True)

# ---------- Verify ----------

print("\nDataset Info:\n")
print(master.info())

print("\nFirst Five Rows:\n")
print(master.head())

print("\nMissing Values After Cleaning:\n")
print(master.isnull().sum())

# ---------- Save ----------

os.makedirs(r"D:\Ai_stock_predictor\data\cleaned", exist_ok=True)

master.to_csv(
    r"D:\Ai_stock_predictor\data\cleaned\cleaned_dataset.csv",
    index=False
)

print("\n✅ Cleaned dataset saved successfully!")