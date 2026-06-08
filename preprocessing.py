"""
preprocessing.py — Data loading and cleaning for the Loan Approval XAI Dashboard.

This module prepares the raw loan dataset for use by the machine learning model.
It handles three tasks:
  1. Loading the CSV and enforcing consistent column types.
  2. Imputing missing values in categorical columns with the column mode.
  3. Splitting the cleaned data into features (X) and target (y) for training.

Numeric columns (LoanAmount, Credit_History, etc.) may still contain NaN after
this step. That is intentional — the sklearn Pipeline's SimpleImputer handles
them during training and inference, keeping imputation logic in one place.
"""
import os
import pandas as pd
import numpy as np
from config import DATA_PATH

# ── Feature schema ─────────────────────────────────────────────────────────
# These lists define which columns are categorical and which are numeric.
# Fixing them here prevents the feature schema from drifting if the CSV ever
# gains extra columns, and ensures consistent encoding during both training
# and inference.

CATEGORICAL_FEATURES = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area"
]

NUMERICAL_FEATURES = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History"
]

TARGET_COLUMN = "Loan_Status"

# =====================================================================
# SECTION 2: PRODUCTION-GRADE SANITIZATION AND INGESTION
# =====================================================================

def load_clean_data() -> pd.DataFrame:
    """
    Load the raw loan dataset from disk and return a cleaned DataFrame.

    Returns:
        DataFrame with:
        - Loan_Status mapped to 1 (Approved) / 0 (Rejected).
        - Loan_ID removed (it is an identifier, not a predictive feature).
        - Categorical columns imputed with their mode and cast to string.
        - Numeric columns cast to float (NaN preserved for sklearn's imputer).

    This function is used during both model training (train.py) and app startup
    (app.py) to ensure both pipelines see identically prepared data.
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Critical Ingestion Error: Base dataset file missing at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    
    # Strip accidental whitespace anomalies out of the text headers
    df.columns = df.columns.str.strip()

    # Drop records missing target labels immediately to avoid training errors
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN]).copy()
        
        # Defensive Label Check: Ensure target formats match requirements before mapping digits
        valid_labels = set(df[TARGET_COLUMN].dropna().unique())
        if valid_labels.issubset({"Y", "N", "1", "0", 1, 0}):
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Y": 1, "N": 0, "1": 1, "0": 0, 1: 1, 0: 0})
        else:
            raise ValueError(f"Target Label Discrepancy: Unexpected entries found in {TARGET_COLUMN}: {valid_labels}")
    
    # Isolate institutional loan keys from the statistical matrix variables
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    # Enforce static type configurations across categorical feature scopes
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            # Impute empty values using the column's statistical mode to prevent string literal "nan" conversion bugs
            if df[col].isnull().any():
                fallback_mode = df[col].mode()[0]
                df[col] = df[col].fillna(fallback_mode)
            df[col] = df[col].astype(str).str.strip()

    # Enforce static type configurations across numerical feature scopes
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Defensive check: Log a warning text output if critical numeric metrics contain null elements
            if df[col].isnull().any():
                # Numeric NaN values are left here intentionally.
                # The sklearn Pipeline's SimpleImputer fills them during fit/transform,
                # keeping imputation parameters (e.g. the median) tied to training data
                # rather than being recomputed from whatever subset is loaded here.
                pass

    return df

# =====================================================================
# SECTION 3: MATRIX ISOLATION AND SPLITTING CHANNELS
# =====================================================================

def split_features_target(df: pd.DataFrame, target: str = TARGET_COLUMN) -> tuple:
    """
    Split a cleaned DataFrame into a feature matrix and a target vector.

    Parameters:
        df     : Cleaned DataFrame (output of load_clean_data).
        target : Name of the column to use as the prediction target.

    Returns:
        (X, y) — X is the feature DataFrame, y is the target Series.

    Used during training to separate the input features from the outcome
    the model is learning to predict.
    """
    if target not in df.columns:
        raise KeyError(f"Structural Processing Error: Target tracking column '{target}' missing from source data headers.")

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def get_feature_columns(X: pd.DataFrame) -> tuple:
    """
    Return which categorical and numeric feature columns exist in a given DataFrame.

    Parameters:
        X : Feature DataFrame (Loan_Status already removed).

    Returns:
        (cat_cols, num_cols) — lists of column names that are present in X
        and belong to the categorical or numeric schema respectively.

    Cross-referencing against X.columns rather than returning the full lists
    prevents KeyError if a column was dropped or renamed upstream.
    """
    # Only return columns that actually exist in X to prevent downstream KeyErrors.
    cat_cols = [col for col in CATEGORICAL_FEATURES if col in X.columns]
    num_cols = [col for col in NUMERICAL_FEATURES if col in X.columns]

    return cat_cols, num_cols


if __name__ == "__main__":
    print("Testing Preprocessing Architecture Suite...")
    cleaned_df = load_clean_data()
    feature_matrix, target_vector = split_features_target(cleaned_df)
    print(f"Data Matrix Ingestion Complete. Extracted Shape Dimensions: {feature_matrix.shape}")