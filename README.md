# Loan Approval XAI Dashboard

A Streamlit dashboard that uses machine learning and explainable AI to support loan approval analysis through predictions, SHAP explanations, similar-case comparison, and what-if decision-support scenarios.

## Features

- Loan approval prediction for selected applications
- Approval probability and decision label
- SHAP-based explanation of model decisions
- Global model insights
- Similar historical case comparison
- What-if / counterfactual scenario suggestions
- Sidebar search and filters
- Educational comments and docstrings for code review

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- SHAP
- Plotly
- joblib
- XGBoost
- DiCE

## Project Structure

```text
.
├── analysis/
│   └── explore.py
├── data/
│   └── LoanApprovalPrediction.csv
├── models/
│   └── best_model.pkl
├── sources/
│   ├── app.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── services/
│   └── ui/
├── .gitignore
├── README.md
└── requirements.txt