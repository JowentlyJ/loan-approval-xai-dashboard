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

## Dataset

This project uses a loan approval prediction dataset based on the example dataset used in the GeeksforGeeks article **“Loan Approval Prediction using Machine Learning.”** The dataset contains historical loan application records with applicant-related features such as gender, marital status, number of dependents, education level, self-employment status, applicant income, co-applicant income, loan amount, loan amount term, credit history, property area, and loan approval status.

In this project, the dataset is used to train and evaluate a machine learning model that predicts whether a loan application is likely to be approved. The same dataset is also used in the dashboard to support model explanations, similar-case comparison, and what-if scenario analysis.

Dataset/article source:  
https://www.geeksforgeeks.org/machine-learning/loan-approval-prediction-using-machine-learning/

> Note: This dataset is used for educational and demonstration purposes. The dashboard should be treated as a decision-support prototype, not as a production-ready financial approval system.

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