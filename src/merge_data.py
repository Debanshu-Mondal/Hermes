import pandas as pd

def load_nifty(file_path):
    df = pd.read_csv(file_path, header=[0,1], index_col=0)

    df.reset_index(inplace=True)

    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df.columns = [
        c[0] if isinstance(c, tuple) else c
        for c in df.columns
    ]

    df = df[["Date","Open","High","Low","Close","Volume"]]

    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)

    return df

def load_data(file_path, column_name):
    df = pd.read_csv(file_path, header=[0,1], index_col=0)

    # Move the index into a column
    df.reset_index(inplace=True)

    # Rename the first column to Date
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Flatten the remaining MultiIndex columns
    df.columns = [
        c[0] if isinstance(c, tuple) else c
        for c in df.columns
    ]

    # Keep Date and Close
    df = df[["Date", "Close"]]

    # Rename Close
    df.rename(columns={"Close": column_name}, inplace=True)

    # Convert Date
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
    return df
# ---------- Load Data ----------

nifty = load_nifty(r"D:\Ai_stock_predictor\data\raw\nifty50.csv")

vix = load_data(r"D:\Ai_stock_predictor\data\raw\india_vix.csv", "VIX")

gold = load_data(r"D:\Ai_stock_predictor\data\raw\gold.csv", "Gold")

crude = load_data(r"D:\Ai_stock_predictor\data\raw\crude.csv", "Crude")

usd = load_data(r"D:\Ai_stock_predictor\data\raw\inr.csv", "USDINR")

sp500 = load_data(r"D:\Ai_stock_predictor\data\raw\s&p.csv", "SP500")

nasdaq = load_data(r"D:\Ai_stock_predictor\data\raw\nasdaq.csv", "NASDAQ")


# ---------- Merge ----------

master = nifty

datasets = [vix, gold, crude, usd, sp500, nasdaq]

for dataset in datasets:
    master = pd.merge(master, dataset, on="Date", how="inner")


# ---------- Sort ----------

master = master.sort_values("Date")

# ---------- Save ----------

master.to_csv(
    r"D:\Ai_stock_predictor\data\processed\master_dataset.csv",
    index=False
)

print("Master dataset created successfully!")
print(master.head())
print(master.info())