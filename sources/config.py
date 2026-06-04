"""
config.py — Central path configuration for the Loan Approval XAI Dashboard.

All file paths used across the project are defined here so that moving
the project folder only requires updating this one file, not every module
that reads data or loads the model.
"""
import os

# Locate the project root by going two levels up from this file's location.
# This makes every path work regardless of which directory the terminal is in.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the raw loan dataset CSV used for loading applicant records.
DATA_PATH = os.path.join(BASE_DIR, "data", "LoanApprovalPrediction.csv")

# Directory where trained model files (.pkl) are saved after training.
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Directory for generated outputs such as SHAP summary plots and model comparison CSVs.
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# The specific model file the app loads at startup — the best-performing
# classifier selected after comparing multiple algorithms during training.
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
