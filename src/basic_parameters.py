import pandas as pd
import numpy as np
import os

# -------------------------------
# Load cleaned dataset
# -------------------------------

df = pd.read_csv(
    r"D:\Ai_stock_predictor\data\cleaned\cleaned_dataset.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

# -------------------------------
# Basic Price Features
# -------------------------------

# Daily Return
df["Daily_Return"] = df["Close"].pct_change()

# Log Return
df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

# Intraday Price Change
df["Open_Close_Pct"] = (
    (df["Close"] - df["Open"]) / df["Open"]
)

# Daily Volatility
df["High_Low_Pct"] = (
    (df["High"] - df["Low"]) / df["Close"]
)

# Price Range
df["Price_Range"] = df["High"] - df["Low"]

# Candle Body Size
df["Body_Size"] = abs(df["Close"] - df["Open"])

# Previous Day Close
df["Prev_Close"] = df["Close"].shift(1)

# Previous Day Return
df["Prev_Return"] = df["Daily_Return"].shift(1)

# 5-Day Return
df["Return_5D"] = (
    df["Close"] / df["Close"].shift(5)
) - 1

# 20-Day Return
df["Return_20D"] = (
    df["Close"] / df["Close"].shift(20)
) - 1

# -------------------------------
# Volume Features
# -------------------------------

# Volume Change
df["Volume_Change"] = df["Volume"].pct_change()

# 5-Day Average Volume
df["Volume_MA5"] = df["Volume"].rolling(5).mean()

# Volume Ratio
df["Volume_Ratio"] = (
    df["Volume"] / df["Volume_MA5"]
)

# -------------------------------
# External Market Returns
# -------------------------------

external = [
    "VIX",
    "Gold",
    "Crude",
    "USDINR",
    "SP500",
    "NASDAQ"
]

for col in external:
    df[f"{col}_Return"] = df[col].pct_change()

# -------------------------------
# Calendar Features
# -------------------------------

df["DayOfWeek"] = df["Date"].dt.dayofweek

df["Month"] = df["Date"].dt.month

df["Quarter"] = df["Date"].dt.quarter

# -------------------------------
# Remove NaN values created by shifts
# -------------------------------

df = df.dropna().reset_index(drop=True)

# -------------------------------
# Save
# -------------------------------

os.makedirs(
    r"D:\Ai_stock_predictor\data\featured",
    exist_ok=True
)

df.to_csv(
    r"D:\Ai_stock_predictor\data\featured\basic_features.csv",
    index=False
)

print("✅ Basic feature engineering completed.")
print(df.head())
print(df.info())