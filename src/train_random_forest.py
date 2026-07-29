import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv(
    r"D:\Ai_stock_predictor\data\labeled\labeled_dataset.csv"
)

print("Dataset Loaded Successfully")
print(df.shape)

# =====================================================
# Convert Labels
# =====================================================

label_mapping = {
    "Sell": 0,
    "Hold": 1,
    "Buy": 2
}

df["Target"] = df["Label"].map(label_mapping)

# =====================================================
# Remove Columns Not Used For Training
# =====================================================

drop_columns = [
    "Date",
    "Label",
    "Target",
    "Future_Close",
    "Future_Return"
]

X = df.drop(columns=drop_columns, errors="ignore")
y = df["Target"]

# =====================================================
# Keep Only Numeric Columns
# =====================================================

X = X.select_dtypes(include=[np.number])

# =====================================================
# Replace Infinity With NaN
# =====================================================

X.replace([np.inf, -np.inf], np.nan, inplace=True)

# =====================================================
# Remove Rows With Missing Values
# =====================================================

valid_rows = ~X.isna().any(axis=1)

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

print("\nDataset after cleaning:", X.shape)

# =====================================================
# Check Again
# =====================================================

print("\nRemaining NaN:", X.isnull().sum().sum())
print("Remaining Inf:", np.isinf(X.values).sum())

# =====================================================
# Train/Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================================
# Random Forest
# =====================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...\n")

model.fit(X_train, y_train)

print("Training Completed!")

# =====================================================
# Prediction
# =====================================================

predictions = model.predict(X_test)

# =====================================================
# Evaluation
# =====================================================

print("\nAccuracy")

accuracy = accuracy_score(y_test, predictions)

print(f"{accuracy:.4f}")

print("\nClassification Report")

print(
    classification_report(
        y_test,
        predictions,
        target_names=["Sell", "Hold", "Buy"]
    )
)

print("\nConfusion Matrix")

print(confusion_matrix(y_test, predictions))

# =====================================================
# Feature Importance
# =====================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features\n")

print(importance.head(20))

# =====================================================
# Save Model
# =====================================================

os.makedirs(
    r"D:\Ai_stock_predictor\models",
    exist_ok=True
)

joblib.dump(
    model,
    r"D:\Ai_stock_predictor\models\random_forest.pkl"
)

importance.to_csv(
    r"D:\Ai_stock_predictor\models\feature_importance.csv",
    index=False
)

print("\nModel Saved Successfully!")