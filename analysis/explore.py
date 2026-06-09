"""
Exploratory script for the Loan Approval dataset. Run this directly to inspect
the raw data: shape, missing values, categorical distributions, numeric summaries,
and a correlation heatmap. Output is printed to the console. This script does not
write any files.
"""
import pandas as pd

# Load dataset
df = pd.read_csv("data/LoanApprovalPrediction.csv")
# Strip whitespace from column names — the CSV has trailing spaces in some headers.
df.columns = df.columns.str.strip()

# Shape, columns, and data types
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nInfo:")
df.info()

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

print("\nMissing values percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))

# Sample rows and duplicates
print("\nSample rows:")
print(df.head())

print("\nDuplicates:", df.duplicated().sum())

# Target variable distribution
print("\nLoan_Status counts:")
print(df["Loan_Status"].value_counts())
print(df["Loan_Status"].value_counts(normalize=True))

# Categorical columns
print("\nCategorical columns:")
for col in df.select_dtypes(include=["object", "category"]).columns:
    print(f"\n{col}")
    print(df[col].value_counts(dropna=False).head(10))

# Numeric columns
num_cols = df.select_dtypes(include=["float64", "int64"]).columns

print("\nNumeric columns summary:")
for num_col in num_cols:
    print(f"\n=== {num_col} ===")
    print("Missing:", df[num_col].isnull().sum())
    print("Min:", df[num_col].min())
    print("Max:", df[num_col].max())
    print("Mean:", df[num_col].mean())
    print("Median:", df[num_col].median())

print("\nCredit_History")
print(df["Credit_History"].value_counts(dropna=False))

print("\nLoan_Amount_Term")
print(df["Loan_Amount_Term"].value_counts(dropna=False).head(10))

print("\nDependents")
print(df["Dependents"].value_counts(dropna=False))

# Imputation check
# Apply the same imputation strategy used in training: median for LoanAmount, mode for others.
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])

print("\nMissing values after filling:")
print(df.isnull().sum())

# Imported here to keep the data inspection above runnable without matplotlib installed.
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation heatmap
numeric_df = df.select_dtypes(include=["float64", "int64"])

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
# Chart is displayed interactively and not saved to disk.
plt.show()
