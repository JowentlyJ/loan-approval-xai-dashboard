"""
prediction.py — Model inference and decision labelling for the Loan Approval XAI Dashboard.

This module is the bridge between applicant profile data and the model's output.
It handles three responsibilities:
  1. Converting raw input into the format the sklearn pipeline expects.
  2. Running the trained model to obtain a probability score.
  3. Translating that probability into a human-readable underwriting decision.

All pages that need a prediction call these functions rather than calling the
model directly, keeping the threshold logic in one place.
"""
import pandas as pd

# ── Decision thresholds ────────────────────────────────────────────────────
# These two cutoffs divide the 0–1 probability range into three zones:
#   >= 0.70         → Approve  (automatic approval)
#   0.45 – 0.69     → Manual Review  (borderline, needs human assessment)
#   < 0.45          → Reject
# Changing either value here propagates to every page in the app at once.
APPROVAL_THRESHOLD = 0.70
REVIEW_THRESHOLD = 0.45

# The complete list of feature columns the model was trained on.
# validate_applicant_data() uses this to catch missing fields before the
# model call, producing a clear error instead of a cryptic sklearn exception.
EXPECTED_COLUMNS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area"
]


def get_decision_label(probability: float) -> str:
    """
    Convert a raw approval probability into a human-readable decision label.

    Parameters:
        probability: Model output between 0 and 1 representing the chance
                     that the loan application will be approved.

    Returns:
        "Approve", "Manual Review", or "Reject" based on the configured thresholds.

    This function is shared by every service and page that needs a decision
    label, ensuring thresholds are applied consistently across the app.
    """
    if probability >= APPROVAL_THRESHOLD:
        return "Approve"
    if probability >= REVIEW_THRESHOLD:
        return "Manual Review"
    return "Reject"


def validate_applicant_data(applicant_data: dict) -> None:
    """
    Check that a raw applicant dictionary contains all required feature columns.

    Parameters:
        applicant_data: Dictionary mapping feature names to their values.

    Raises:
        ValueError: If any expected column is absent from the dictionary.

    This guard prevents obscure sklearn errors that would otherwise appear
    deep inside the pipeline when a required column is missing.
    """
    missing_columns = set(EXPECTED_COLUMNS) - set(applicant_data.keys())
    if missing_columns:
        raise ValueError(f"Missing applicant fields: {missing_columns}")


def predict_applicant(model, applicant_features: pd.DataFrame):
    """
    Run the trained model pipeline on a single-row applicant DataFrame.

    Parameters:
        model             : Trained sklearn Pipeline (preprocessor + classifier).
        applicant_features: One-row DataFrame containing only feature columns.
                            Loan_Status must be removed before calling this function.

    Returns:
        (prediction, probability, decision)
        prediction  : int   — 1 for approved, 0 for rejected.
        probability : float — approval probability between 0 and 1.
        decision    : str   — "Approve", "Manual Review", or "Reject".

    The sklearn pipeline internally handles encoding and imputation before
    reaching the classifier, so the input does not need to be pre-processed.
    """
    prediction = int(model.predict(applicant_features)[0])
    # predict_proba returns [prob_class_0, prob_class_1]; index [1] is the
    # probability of approval (class 1).
    probability = float(model.predict_proba(applicant_features)[0][1])
    decision = get_decision_label(probability)

    return prediction, probability, decision


def predict_applicant_from_dict(model, applicant_data: dict) -> dict:
    """
    Validate and predict from a raw dictionary of applicant values.

    Parameters:
        model          : Trained sklearn Pipeline.
        applicant_data : Plain dictionary mapping feature names to values,
                         e.g. {"Gender": "Male", "LoanAmount": 150, ...}.

    Returns:
        Dictionary with keys "prediction", "probability", and "decision".

    Streamlit collects form values one by one, so this function accepts a
    plain dictionary and converts it into the single-row DataFrame the model
    expects. Validation runs first so any missing fields raise a clear error
    before the model is called.
    """
    validate_applicant_data(applicant_data)
    # The model expects a table (DataFrame), not a bare dictionary.
    # Wrapping the dict in a list and passing it to DataFrame creates one row.
    applicant_df = pd.DataFrame([applicant_data])
    prediction, probability, decision = predict_applicant(model, applicant_df)

    return {
        "prediction": prediction,
        "probability": probability,
        "decision": decision,
    }
