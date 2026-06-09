import os
import time
from datetime import datetime
import pandas as pd
import streamlit as st

from config import OUTPUT_DIR

ASSESSMENTS_PATH = os.path.join(OUTPUT_DIR, "underwriter_assessments.csv")
ASSESSMENT_COLUMNS = [
    "applicant_key", "loan_id", "original_row", "underwriter_name",
    "model_probability", "model_decision", "underwriter_decision",
    "risk_level", "income_verified", "employment_verified",
    "credit_reviewed", "documents_complete", "fraud_concern",
    "conditions", "notes", "saved_at"
]

# =====================================================================
# SECTION 1: Load and save underwriter assessment records
# =====================================================================
# This section handles reading from and writing to the assessments CSV,
# with a basic retry mechanism to handle simultaneous saves.

def load_assessments() -> pd.DataFrame:
    """Load all saved underwriter assessments from disk, or return an empty DataFrame."""
    if os.path.exists(ASSESSMENTS_PATH):
        try:
            return pd.read_csv(ASSESSMENTS_PATH)
        except Exception:
            time.sleep(0.1)  # Fallback sleep interval if file is locked by another thread
            return pd.read_csv(ASSESSMENTS_PATH)
    return pd.DataFrame(columns=ASSESSMENT_COLUMNS)


def save_assessment_record(record: dict) -> None:
    """
    Save a new assessment record to disk, replacing any existing record for the same applicant.

    Retries up to three times if the file is temporarily locked during concurrent saves.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Retry loop acting as a basic file locking mechanism for multi-user environments
    for attempt in range(3):
        try:
            assessments_df = load_assessments()
            applicant_key = record["applicant_key"]

            if not assessments_df.empty and applicant_key in assessments_df["applicant_key"].values:
                assessments_df = assessments_df[assessments_df["applicant_key"] != applicant_key]

            updated_df = pd.concat([assessments_df, pd.DataFrame([record])], ignore_index=True)
            updated_df.to_csv(ASSESSMENTS_PATH, index=False)
            break
        except IOError:
            time.sleep(0.2)  # Delay execution to let competing system operations clear


def get_applicant_key(applicant_display: pd.Series) -> str:
    """Build a unique string key for an applicant using their Loan ID or row index."""
    loan_id = applicant_display.get("Loan_ID", None)
    original_row = applicant_display.get("Original_Row", None)
    if pd.notna(loan_id):
        return f"loan_id::{loan_id}"
    return f"row::{original_row}"


def get_existing_assessment(applicant_display: pd.Series):
    """Return the most recent saved assessment for an applicant, or None if none exists."""
    assessments_df = load_assessments()
    if assessments_df.empty:
        return None

    applicant_key = get_applicant_key(applicant_display)
    matching_rows = assessments_df[assessments_df["applicant_key"] == applicant_key]
    if matching_rows.empty:
        return None
    return matching_rows.iloc[-1].to_dict()


def safe_text(value, fallback: str = "") -> str:
    """Return fallback if value is NaN, otherwise return value as a string."""
    return fallback if pd.isna(value) else str(value)


def get_selectbox_index(options: list, selected_value: str, fallback: str) -> int:
    """Return the index of selected_value in options, or the index of fallback if not found."""
    if selected_value in options:
        return options.index(selected_value)
    return options.index(fallback)


# =====================================================================
# SECTION 2: Render the underwriter assessment form
# =====================================================================
# This section displays the manual sign-off form where an underwriter can
# record their decision, risk level, and notes for the selected applicant.

def render_underwriter_assessment_section(applicant_display: pd.Series, probability: float, model_decision: str) -> None:
    """
    Render the underwriter assessment form for the selected applicant.

    Loads any existing assessment for this applicant and pre-fills the form,
    then saves the result when the underwriter submits.
    """
    st.subheader("Underwriter Assessment")
    existing = get_existing_assessment(applicant_display)

    default_underwriter_name = safe_text(existing.get("underwriter_name"), "") if existing else ""
    default_underwriter_decision = safe_text(existing.get("underwriter_decision"), model_decision) if existing else model_decision
    default_risk_level = safe_text(existing.get("risk_level"), "Medium") if existing else "Medium"
    default_conditions = safe_text(existing.get("conditions"), "") if existing else ""
    default_notes = safe_text(existing.get("notes"), "") if existing else ""

    decision_options = ["Approve", "Manual Review", "Reject"]
    risk_options = ["Low", "Medium", "High"]

    with st.form("underwriter_assessment_form"):
        st.write("Record manual review adjustments and credit parameters here.")
        
        underwriter_name = st.text_input("Underwriter Name", value=default_underwriter_name)
        underwriter_decision = st.selectbox(
            "Final Underwriter Decision", 
            decision_options, 
            index=get_selectbox_index(decision_options, default_underwriter_decision, model_decision)
        )
        risk_level = st.selectbox(
            "Risk Level", 
            risk_options, 
            index=get_selectbox_index(risk_options, default_risk_level, "Medium")
        )
        conditions = st.text_area("Approval Conditions / Follow-up Requirements", value=default_conditions, height=100)
        notes = st.text_area("Underwriter Notes", value=default_notes, height=140)

        submitted = st.form_submit_button("Save Assessment")

    if submitted:
        record = {
            "applicant_key": get_applicant_key(applicant_display),
            "loan_id": applicant_display.get("Loan_ID", ""),
            "original_row": applicant_display.get("Original_Row", ""),
            "underwriter_name": underwriter_name,
            "model_probability": round(probability, 6),
            "model_decision": model_decision,
            "underwriter_decision": underwriter_decision,
            "risk_level": risk_level,
            "income_verified": "", "employment_verified": "", "credit_reviewed": "",
            "documents_complete": "", "fraud_concern": "",
            "conditions": conditions, "notes": notes,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_assessment_record(record)
        st.success("Underwriter assessment logged successfully.")
        st.rerun()  # Forces immediate sync of historical changes across dashboard containers

    if existing:
        st.caption("🚨 Notice: An active historical assessment log was located and loaded for this applicant profile.")