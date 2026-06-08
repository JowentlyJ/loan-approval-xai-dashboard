# Loan Approval XAI Dashboard

A Streamlit dashboard that uses machine learning and explainable AI (XAI) to support loan approval analysis through predictions, feature contribution explanations (SHAP), similar-case comparison, fairness metrics, and what-if decision-support scenarios.

**Live demo:** [Loan Approval XAI Dashboard](https://loan-approval-xai-dashboard-jjush6b8w5f5yjcz6zzkt7.streamlit.app/)
---

## Metadata

| Field                     | Description                                                                         |
| ------------------------- | ----------------------------------------------------------------------------------- |
| Project name              | Loan Approval XAI Dashboard                                                         |
| Project context           | FIN-X Research Group                                                                |
| Author                    | Jowently Josephina                                                                  |
| Domain                    | Financial services / Loan underwriting                                              |
| Project type              | Data-driven application / Explainable AI prototype                                  |
| Main framework            | Streamlit                                                                           |
| Programming language      | Python                                                                              |
| Machine-learning task     | Binary classification                                                               |
| Target variable           | `Loan_Status`                                                                       |
| Output classes            | Approved / Rejected                                                                 |
| Final selected model      | Logistic Regression                                                                 |
| Initial best model        | XGBoost                                                                             |
| Evaluation method         | 5-Fold Stratified Cross-Validation                                                  |
| Explainability methods    | SHAP, similar-case comparison, What-If / counterfactual scenarios                   |
| Fairness metrics          | Disparate Impact Ratio, Demographic Parity Difference, Equal Opportunity Difference |
| Human-in-the-Loop feature | Manual final decision and written rationale field                                   |
| Status                    | Educational prototype / research prototype                                          |

---

## Overview

This project provides a loan approval decision-support dashboard built with Streamlit. The dashboard allows users to inspect loan applications, view predicted approval outcomes, understand model reasoning, compare similar historical cases, and explore what-if scenarios.

The goal is not to automate final loan approval decisions. Instead, the dashboard supports underwriters by making machine-learning outputs more transparent and easier to question. The user remains responsible for the final decision.

The system combines:

* Machine-learning prediction
* Feature contribution explanation (SHAP)
* Similar historical case comparison
* What-if / counterfactual scenario exploration
* Fairness evaluation
* Human-in-the-loop decision support

---

## Features

* Loan approval prediction for selected applications
* Approval probability and decision label
* Feature contribution explanation (SHAP) of individual model decisions
* Global model insights
* Similar historical case comparison
* What-if / counterfactual scenario suggestions
* Fairness metric overview
* Sidebar search and filters
* Manual final decision field
* Written rationale input
* Educational comments and docstrings for code review

---

## Dataset

This project uses a loan approval prediction dataset based on the example dataset used in the GeeksforGeeks article **“Loan Approval Prediction using Machine Learning.”**

Dataset/article source:
https://www.geeksforgeeks.org/machine-learning/loan-approval-prediction-using-machine-learning/

The dataset contains historical loan application records with applicant-related features such as:

* Gender
* Marital status
* Number of dependents
* Education level
* Self-employment status
* Applicant income
* Co-applicant income
* Loan amount
* Loan amount term
* Credit history
* Property area
* Loan approval status

The dataset contains **598 applications** and **13 columns**. The target variable is `Loan_Status`, where approved applications are labelled as `Y` and rejected applications are labelled as `N`.

The target distribution is imbalanced:

| Class    | Count | Percentage |
| -------- | ----: | ---------: |
| Approved |   411 |      68.7% |
| Rejected |   187 |      31.3% |

This imbalance is important because the model may become better at predicting approvals than rejections.

> Note: This dataset is used for educational and demonstration purposes. The dashboard should be treated as a decision-support prototype, not as a production-ready financial approval system.

---

## Tech Stack

* Python
* Streamlit
* pandas
* NumPy
* scikit-learn
* SHAP
* Plotly
* joblib
* XGBoost
* DiCE

---

## Model Development

Four machine-learning models were compared:

| Model               | Initial Accuracy | Initial ROC-AUC score |
| ------------------- | ---------------: | --------------: |
| Logistic Regression |           0.7833 |          0.8341 |
| Random Forest       |           0.8167 |          0.8245 |
| Gradient Boosting   |           0.8000 |          0.8498 |
| XGBoost             |           0.8167 |          0.8642 |

In the first evaluation using a single train/test split, XGBoost performed best with an accuracy of **0.8167** and a ROC-AUC score of **0.8642**.

However, because the dataset is relatively small and imbalanced, a single split can produce unstable results. The evaluation was therefore improved using **5-Fold Stratified Cross-Validation**.

| Model               | Cross-validated Accuracy | Cross-validated ROC-AUC score |
| ------------------- | -----------------------: | ----------------------: |
| Logistic Regression |                   80.94% |                  0.7602 |
| Random Forest       |                   80.43% |                  0.7570 |
| XGBoost             |                   77.59% |                  0.7418 |
| Gradient Boosting   |                   76.59% |                  0.7305 |

Based on cross-validation, **Logistic Regression** was selected as the final model because it generalized better and is more interpretable for a decision-support dashboard.

---

## Explainability Layer

The dashboard includes multiple explanation components.

### Feature Contribution Explanations (SHAP)

SHAP is a feature contribution analysis method that shows how individual applicant details influenced the model's prediction. Each feature receives a score that indicates how strongly it pushed the outcome toward approval or rejection.

### Similar Case Comparison

The similar case comparison feature shows historical applications that are similar to the current applicant. This helps users compare the current case with previous approved and rejected applications.

### What-If / Counterfactual Scenarios

The What-If tool allows users to explore how changing applicant variables could affect the prediction. These scenarios should be interpreted as hypothetical model-based outputs, not as guaranteed recommendations.

---

## Fairness Evaluation

Fairness was evaluated because the dataset contains demographic and socio-economic variables that may reflect historical bias or proxy relationships.

The final Logistic Regression model was evaluated using:

| Metric                        | Result | Meaning                                           |
| ----------------------------- | -----: | ------------------------------------------------- |
| Disparate Impact Ratio        | 0.9761 | Compares lowest and highest group selection rates |
| Demographic Parity Difference | 0.0204 | Measures gap in predicted approval rates          |
| Equal Opportunity Difference  | 0.0043 | Measures gap in true positive rates               |

These results suggest low group-level disparity in this prototype evaluation. However, fairness metrics do not prove that the model is fully fair. They should be treated as diagnostic indicators.

---

## Human-in-the-Loop Workflow

The dashboard is designed so that the model does not make the final decision. The user remains responsible for the final assessment.

Workflow:

1. Select or search for a loan application.
2. Review applicant information.
3. View the model prediction and approval probability.
4. Inspect feature contribution explanations.
5. Compare similar historical cases.
6. Explore What-If scenarios.
7. Record a final decision.
8. Write a professional rationale.

This workflow supports accountability and reduces the risk of treating the model as the final authority.

---

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
```

---

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd <your-project-folder>
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run sources/app.py
```

---

## Usage

After starting the Streamlit app, use the sidebar to select or search for a loan application. The dashboard will display the predicted loan outcome, approval probability, and explanation components.

Users can:

* Inspect the predicted decision
* Review feature contribution explanations
* Compare the applicant with similar historical cases
* Explore What-If scenarios
* Record a final decision and rationale

---

## Limitations

This project is an educational prototype and has several limitations:

* The dataset is relatively small.
* The target variable is imbalanced.
* The dataset may contain historical bias.
* Fairness metrics only measure selected group-level patterns.
* SHAP explanations may be difficult for non-expert users.
* Counterfactual outputs may be misread as recommendations.
* The prototype does not include soft data such as market context, client history, temporary hardship, or professional judgement.

---

## Ethical Considerations

This dashboard should not be used as an automated loan approval system. It is intended to support human judgement, not replace it.

Important ethical considerations include:

* Avoiding blind reliance on model outputs
* Keeping the underwriter responsible for the final decision
* Communicating model limitations clearly
* Monitoring bias and fairness over time
* Treating explanations as model-based interpretations, not complete truths
* Ensuring that applicants are not reduced only to data profiles

---

## Future Improvements

Future work could include:

* Testing with professional underwriters
* Improving SHAP labels and textual explanations
* Adding clearer warnings for What-If outputs
* Expanding fairness analysis
* Adding model drift monitoring
* Logging underwriter disagreements with the model
* Creating a feedback loop for model improvement
* Adding model cards and dataset documentation
* Including support for soft contextual data

---

## License

This project is intended for educational and research purposes.