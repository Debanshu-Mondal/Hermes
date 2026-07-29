import pandas as pd
import ta
import os

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv(
    r"D:\Ai_stock_predictor\data\featured\basic_features.csv"
)

# -----------------------------
# Simple Moving Averages
# -----------------------------

df["SMA_5"] = ta.trend.sma_indicator(df["Close"], window=5)

df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)

df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)

df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)

# -----------------------------
# Exponential Moving Averages
# -----------------------------

df["EMA_10"] = ta.trend.ema_indicator(df["Close"], window=10)

df["EMA_20"] = ta.trend.ema_indicator(df["Close"], window=20)

df["EMA_50"] = ta.trend.ema_indicator(df["Close"], window=50)

# -----------------------------
# RSI
# -----------------------------

df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

# -----------------------------
# MACD
# -----------------------------

macd = ta.trend.MACD(df["Close"])

df["MACD"] = macd.macd()

df["MACD_Signal"] = macd.macd_signal()

df["MACD_Hist"] = macd.macd_diff()

# -----------------------------
# Bollinger Bands
# -----------------------------

bollinger = ta.volatility.BollingerBands(df["Close"])

df["BB_High"] = bollinger.bollinger_hband()

df["BB_Low"] = bollinger.bollinger_lband()

df["BB_Middle"] = bollinger.bollinger_mavg()

df["BB_Width"] = bollinger.bollinger_wband()

# -----------------------------
# ATR
# -----------------------------

df["ATR"] = ta.volatility.average_true_range(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
)

# -----------------------------
# ADX
# -----------------------------

df["ADX"] = ta.trend.adx(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
)

# -----------------------------
# CCI
# -----------------------------

df["CCI"] = ta.trend.cci(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=20
)

# -----------------------------
# ROC
# -----------------------------

df["ROC"] = ta.momentum.roc(df["Close"], window=12)

# -----------------------------
# Stochastic Oscillator
# -----------------------------

stoch = ta.momentum.StochasticOscillator(
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
)

df["Stoch_K"] = stoch.stoch()

df["Stoch_D"] = stoch.stoch_signal()

# -----------------------------
# Williams %R
# -----------------------------

df["Williams_R"] = ta.momentum.williams_r(
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
)

# -----------------------------
# OBV
# -----------------------------

df["OBV"] = ta.volume.on_balance_volume(
    close=df["Close"],
    volume=df["Volume"]
)

# -----------------------------
# Chaikin Money Flow
# -----------------------------

df["CMF"] = ta.volume.chaikin_money_flow(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    volume=df["Volume"]
)

# -----------------------------
# Drop NaN
# -----------------------------

df = df.dropna().reset_index(drop=True)

# -----------------------------
# Save
# -----------------------------

os.makedirs(
    r"D:\Ai_stock_predictor\data\featured",
    exist_ok=True
)

df.to_csv(
    r"D:\Ai_stock_predictor\data\featured\technical_features.csv",
    index=False
)

print("Technical indicators added successfully!")
print(df.head())
print(df.info())