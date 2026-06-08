"""
app.py — Main entry point for the Loan Approval XAI Dashboard.

This file does four things every time Streamlit reruns the script:
  1. Loads the trained model and dataset into memory (cached after first run).
  2. Builds the sidebar: search, applicant dropdown, navigation, and filters.
  3. Applies filters to produce a subset of applicants for the dropdown.
  4. Routes to the correct page function based on the selected navigation item.

Data flow overview:
  - User picks filters → mask is computed → filtered_options built → dropdown shown.
  - User selects applicant → current_index (original label) stored in session state.
  - Page function receives current_index and uses it to look up the applicant in
    both display_df (for readable values) and model_df (for prediction / SHAP).

Key index concept:
  display_df has a RangeIndex (0, 1, 2, …). model_df is a subset with the same
  labels but gaps where rows had missing Loan_Status. current_index is always a
  label from the original RangeIndex, so display_df.iloc[i] and model_df.loc[[i]]
  both resolve to the same real-world applicant.
"""
import os
import joblib
import pandas as pd
import streamlit as st
import numpy as np

from config import DATA_PATH, BEST_MODEL_PATH, OUTPUT_DIR

# Frontend Page Component Routines
from ui.loan_decision_page import render_loan_decision_page
from ui.model_insights_page import render_model_insights_page
from ui.similar_cases_page import render_similar_cases_page
from ui.counterfactual_page import render_counterfactual_page

# Core Analytical Core Integration Blocks
from services.prediction import predict_applicant
from services.assessment import render_underwriter_assessment_section
from services.explainability import (
    get_local_shap_values,
    create_local_shap_bar_chart,
)
from services.counterfactual import (
    generate_counterfactuals_for_approval,
    render_dice_explanation_plot,
)
from services.similarity import (
    get_similar_cases,
    get_similar_cases_by_class,
)

# Administrative Configuration Controls
GLOBAL_SHAP_IMAGE = os.path.join(OUTPUT_DIR, "global_shap_summary.png")

# =====================================================================
# SECTION 1: SYSTEM-WIDE PAGE FRAMEWORK INITIALIZATION
# =====================================================================

st.set_page_config(
    page_title="Loan Approval XAI Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Sidebar nav: radio items styled as clickable cards */
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    display: flex;
    align-items: center;
    width: 100%;
    padding: 9px 12px;
    margin: 2px 0;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.02);
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    transition: background 0.15s, border-color 0.15s;
    box-sizing: border-box;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(37,99,235,0.08);
    border-color: rgba(37,99,235,0.3);
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(37,99,235,0.12);
    border-color: rgba(37,99,235,0.7);
    font-weight: 700;
}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    display: none !important;
}
section[data-testid="stSidebar"] .stRadio {
    width: 100%;
}
section[data-testid="stSidebar"] div[role="radiogroup"] {
    width: 100%;
}
section[data-testid="stSidebar"] .element-container:has(.stRadio) {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# SECTION 2: DEFENSIVE ASSET LOADING ENGINES
# =====================================================================

@st.cache_resource
def load_production_model():
    """
    Load the trained sklearn Pipeline from disk and cache it for the session.

    Returns:
        Fitted sklearn Pipeline containing the preprocessor and classifier.

    @st.cache_resource means this runs once per session and the model object
    is reused on every page rerun, avoiding the cost of deserializing a large
    pickle file on every user interaction.
    """
    try:
        if not os.path.exists(BEST_MODEL_PATH):
            st.error(f"⚠️ Deployment Failure: Production model file not found at path location: {BEST_MODEL_PATH}")
            st.stop()
        return joblib.load(BEST_MODEL_PATH)
    except Exception as error:
        st.error(f"🚨 Model Deserialization Critical Error: Corrupted architecture file. Details: {error}")
        st.stop()


@st.cache_data
def load_production_dataset() -> tuple:
    """
    Load the loan CSV and return two DataFrames for different purposes.

    Returns:
        display_df : Full dataset with raw readable values used in UI tables.
                     Includes an "Original_Row" column for traceability.
        model_df   : Cleaned subset used for prediction and SHAP — rows with
                     missing Loan_Status are dropped, Loan_ID is removed,
                     and Loan_Status is encoded as 1/0.

    Two separate DataFrames are needed because the user-facing tables should
    show human-readable values (e.g. "Y"/"N" for loan status, Loan_ID for
    identification), while the model requires numeric encoding and no ID columns.
    Both DataFrames preserve the original row index from raw_df, so a label
    from one can safely look up the same applicant in the other.
    """
    try:
        if not os.path.exists(DATA_PATH):
            st.error(f"⚠️ Ingestion Failure: Target portfolio source database not found at: {DATA_PATH}")
            st.stop()

        raw_df = pd.read_csv(DATA_PATH)
        raw_df.columns = raw_df.columns.str.strip()  # Remove accidental whitespace from column headers.

        # display_df keeps every raw column so the UI can show full readable details.
        # Original_Row is inserted so tables always show which CSV row each applicant came from.
        display_df = raw_df.copy()
        display_df.insert(0, "Original_Row", range(len(display_df)))

        # model_df drops rows without a known outcome — they cannot be used for
        # training or SHAP explanation because the target label is unknown.
        model_df = raw_df.dropna(subset=["Loan_Status"]).copy()
        
        if "Loan_ID" in model_df.columns:
            model_df = model_df.drop(columns=["Loan_ID"])
            
        model_df["Loan_Status"] = model_df["Loan_Status"].map({"Y": 1, "N": 0})
        
        if "Dependents" in model_df.columns:
            model_df["Dependents"] = model_df["Dependents"].fillna(model_df["Dependents"].mode()[0]).astype(str)

        return display_df, model_df
    except Exception as error:
        st.error(f"🚨 Database Compilation Pipeline Fault: {error}")
        st.stop()


# Run secure asset validation arrays
model = load_production_model()
display_df, model_df = load_production_dataset()


@st.cache_data
def compute_all_decisions(_model, _model_df):
    """
    Pre-compute the predicted decision label for every applicant in model_df.

    Returns:
        Series indexed by model_df's labels, values are "Approve", "Manual Review",
        or "Reject" strings.

    This result is used by the "Prediction Outcome" sidebar filter. Computing it
    once and caching it avoids running predict_proba on every Streamlit rerun,
    which would be slow for large datasets. The cached Series is then queried
    per-row during filter mask construction.
    """
    X = _model_df.drop(columns=["Loan_Status"], errors="ignore")
    probs = _model.predict_proba(X)[:, 1]
    decisions = [
        "Approve" if p >= 0.70 else ("Manual Review" if p >= 0.45 else "Reject")
        for p in probs
    ]
    return pd.Series(decisions, index=_model_df.index, name="decision")


all_decisions = compute_all_decisions(model, model_df)

# =====================================================================
# SECTION 3: WORKFLOW STATE LIFECYCLE MANAGEMENT
# =====================================================================

# EXPLANATION SECTION: Initialize the unified session state memory index.
# This prevents selection reset loops when switching dashboard analytical windows.
if "selected_applicant_index" not in st.session_state:
    st.session_state["selected_applicant_index"] = 0
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Loan Decision Core"

# =====================================================================
# SECTION 4: GLOBAL SIDEBAR NAVIGATION LAYOUT
# =====================================================================

st.sidebar.title("Loan Approval XAI Dashboard")

st.sidebar.markdown(
    """
    <style>
    .sb-overview {
        font-size: 12px;
        color: rgba(180,180,200,0.85);
        margin: 2px 0 10px 0;
        line-height: 1.5;
    }
    .project-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-bottom: 6px;
    }
    .project-tags span {
        font-size: 10.5px;
        padding: 2px 8px;
        border-radius: 6px;
        background: rgba(37,99,235,0.1);
        border: 1px solid rgba(37,99,235,0.28);
        color: rgba(147,197,253,0.95);
        white-space: nowrap;
    }
    </style>
    <p class="sb-overview">
        Machine learning dashboard for loan approval analysis, model explanations,
        similar-case comparison, and what-if decision support.
    </p>
    <div class="project-tags">
        <span>Python</span>
        <span>Streamlit</span>
        <span>Scikit-learn</span>
        <span>SHAP</span>
        <span>Machine Learning</span>
        <span>Explainable AI</span>
        <span>Loan Approval</span>
        <span>Decision Support</span>
        <span>FinTech</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# ── Visual containers defined in desired display order ─────────────────
# Streamlit renders widgets in the order they appear in code, but the
# dropdown must only show applications that match the current filters —
# which means filters must run first in code even though they appear below
# the dropdown visually. st.sidebar.container() solves this by reserving a
# visual slot at the point of creation; content is written to the slot later,
# in logical order (filters → mask → dropdown → nav).
_c_search   = st.sidebar.container()   # slot 1 – Search Loan ID
_c_dropdown = st.sidebar.container()   # slot 2 – Select Loan Application ID (filled after mask)
st.sidebar.markdown("---")
_c_nav      = st.sidebar.container()   # slot 3 – Navigation radio
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
_c_filters  = st.sidebar.container()   # slot 4 – Filter dropdowns / slider

# ── Fill slot 4 first: filter values are needed to build the mask ───────
with _c_filters:
    income_col = display_df["ApplicantIncome"].dropna()
    income_min, income_max = int(income_col.min()), int(income_col.max())
    if income_min < income_max:
        income_range = st.slider(
            "Applicant Income", income_min, income_max,
            (income_min, income_max), step=500, key="filter_income",
        )
    else:
        income_range = (income_min, income_max)
    decision_filter = st.selectbox(
        "Prediction Outcome", ["All", "Approve", "Manual Review", "Reject"],
        key="filter_decision",
    )
    status_filter = st.selectbox(
        "Actual Loan Status", ["All", "Approved (Y)", "Rejected (N)"],
        key="filter_status",
    )
    credit_filter = st.selectbox(
        "Credit History", ["All", "Has credit history", "No credit history"],
        key="filter_credit",
    )

# ── Fill slot 1: search input (value also feeds the mask) ─────────────
with _c_search:
    st.markdown("### Search Loan ID")
    loan_id_search = ""
    if "Loan_ID" in display_df.columns:
        loan_id_search = st.text_input(
            "Search Loan ID", "", key="search_loan_id",
            placeholder="Search by Loan ID, e.g. LP001002",
            label_visibility="collapsed",
        )
        st.caption("Type a full or partial Loan ID, for example: LP001002")

# ── Apply all filters to build a boolean mask over display_df ──────────
# Each active filter ANDs another condition onto the mask. Rows where the
# mask is False are hidden from the dropdown. The original DataFrame index
# is preserved throughout — no reset_index() — so every row that survives
# still carries its original label, which is what model_df.loc[] needs.
mask = pd.Series(True, index=display_df.index)

if decision_filter != "All":
    decision_mask = pd.Series(
        [all_decisions.get(i, "N/A") == decision_filter for i in display_df.index],
        index=display_df.index,
    )
    mask = mask & decision_mask

if status_filter != "All":
    status_val = "Y" if "Approved" in status_filter else "N"
    if "Loan_Status" in display_df.columns:
        mask = mask & (display_df["Loan_Status"].fillna("") == status_val)

if credit_filter != "All":
    ch_val = 1.0 if credit_filter == "Has credit history" else 0.0
    if "Credit_History" in display_df.columns:
        mask = mask & (display_df["Credit_History"] == ch_val)

mask = mask & (
    (display_df["ApplicantIncome"] >= income_range[0])
    & (display_df["ApplicantIncome"] <= income_range[1])
)

if loan_id_search.strip() and "Loan_ID" in display_df.columns:
    mask = mask & display_df["Loan_ID"].str.contains(
        loan_id_search.strip(), case=False, na=False
    )

filtered_display_df = display_df[mask]

if filtered_display_df.empty:
    st.sidebar.warning("No applications match the filters.")
    st.warning("No applications match the selected filters. Adjust the criteria in the sidebar.")
    st.stop()

# ── Build parallel structures for the dropdown ─────────────────────────
# filtered_indices and filtered_options are kept in sync: position i in
# filtered_options corresponds to position i in filtered_indices.
# This lets on_applicant_change() translate a selected option string back
# to the original DataFrame label without scanning display_df.
filtered_indices = list(filtered_display_df.index)  # original integer labels
filtered_options = []                                # matching human-readable strings
for orig_idx in filtered_indices:
    row = display_df.loc[orig_idx]
    loan_id = str(row.get("Loan_ID", f"Index-{orig_idx}"))
    income = f"${int(row.get('ApplicantIncome', 0)):,}"
    filtered_options.append(f"{loan_id} (Income: {income})")

# If the previously selected applicant is no longer in the filtered set
# (e.g. the user tightened a filter), reset to the first matching row.
if st.session_state["selected_applicant_index"] not in filtered_indices:
    st.session_state["selected_applicant_index"] = filtered_indices[0]

selected_filtered_pos = filtered_indices.index(st.session_state["selected_applicant_index"])


def on_applicant_change():
    """
    Callback fired when the user changes the applicant dropdown.

    Translates the selected option string back to the original DataFrame label
    and stores it in session state so all pages read the same applicant.
    """
    selected_label = st.session_state["sidebar_selector_widget"]
    pos = filtered_options.index(selected_label)
    st.session_state["selected_applicant_index"] = filtered_indices[pos]


# ── Fill slot 2: applicant dropdown (uses filtered_options from above) ──
with _c_dropdown:
    selected_option = st.selectbox(
        "Select Loan Application ID:",
        options=filtered_options,
        index=selected_filtered_pos,
        key="sidebar_selector_widget",
        on_change=on_applicant_change,
    )

# Extract current position values for child sub-page layouts (original iloc = label on RangeIndex)
current_index = st.session_state["selected_applicant_index"]

# ── Fill slot 3: navigation radio ──────────────────────────────────────
_NAV_KEYS = ["Loan Decision Core", "Macro Model Insights", "Peer Similarity Cases", "What-If Counterfactuals"]
_NAV_LABELS = {
    "Loan Decision Core":      "⚖️ Loan Decision",
    "Macro Model Insights":    "📊 Model Insights",
    "Peer Similarity Cases":   "🔍 Similar Cases",
    "What-If Counterfactuals": "🔁 Counterfactuals",
}
_nav_default = _NAV_KEYS.index(st.session_state["active_page"])
with _c_nav:
    st.markdown("### Features")
    page = st.radio(
        "Navigation",
        _NAV_KEYS,
        index=_nav_default,
        format_func=lambda k: _NAV_LABELS[k],
        label_visibility="collapsed",
    )
st.session_state["active_page"] = page

def structural_applicant_selector(*args, **kwargs):
    """
    Return the currently selected applicant's original label index.

    Each page function receives this as its applicant_selector argument.
    They call it to get current_index and then look up the applicant using
    display_df.iloc[index] or model_df.loc[[index]].
    """
    return current_index


def format_single_applicant_display(pos):
    """
    Resolve a flexible applicant reference to a single display_df row (Series).

    Parameters:
        pos: The applicant reference — can be an integer index, a digit string,
             a Loan_ID string (e.g. "LP001002"), a combined dropdown string
             (e.g. "LP001002 (Income: $5,849)"), a pandas Series, or a DataFrame.

    Returns:
        A pandas Series representing one applicant's display row.

    Page functions receive applicant data in different forms depending on context,
    so this function normalises all of them to a single Series before display.
    """
    # If it's already a Series (e.g. passed directly from display_df.iloc[i])
    if isinstance(pos, pd.Series):
        return pos
    if isinstance(pos, pd.DataFrame):
        return pos.iloc[0]

    # Integer or numeric string → positional lookup
    if isinstance(pos, (int, np.integer)) or (isinstance(pos, str) and pos.isdigit()):
        return display_df.iloc[int(pos)]

    # Loan_ID string, possibly with a trailing " (Income: ...)" suffix from the dropdown
    if isinstance(pos, str):
        clean_id = pos.split(" ")[0].strip()
        if "Loan_ID" in display_df.columns:
            matching_rows = display_df[display_df["Loan_ID"] == clean_id]
            if not matching_rows.empty:
                return matching_rows.iloc[0]

    # Fallback to the session-state tracked selection if nothing else matched
    return display_df.iloc[st.session_state["selected_applicant_index"]]


# ── Page router ────────────────────────────────────────────────────────
# Each branch calls one render_* function, passing the model, both DataFrames,
# and the two helper functions (applicant_selector, format_single_applicant_display).
# Service functions (predict, SHAP, similarity, counterfactual) are also passed
# so pages do not import services directly — keeping the UI and service layers decoupled.

if page == "Loan Decision Core":
    render_loan_decision_page(
        model,
        display_df,
        model_df,
        structural_applicant_selector,
        format_single_applicant_display,
        predict_applicant,
        None,  # Handles deprecated charts
        None,
        None,
        get_local_shap_values,
        create_local_shap_bar_chart,
        render_underwriter_assessment_section,
    )

elif page == "Macro Model Insights":
    from services.explainability import (
        calculate_global_shap_importance,
        plot_global_feature_importance
    )
    
    # Custom dashboard mapping routine using memory subplots
    def get_model_comparison_df():
        if os.path.exists(os.path.join(OUTPUT_DIR, "model_performance_cv.csv")):
            return pd.read_csv(os.path.join(OUTPUT_DIR, "model_performance_cv.csv"))
        return pd.DataFrame({"Notice": ["Run train.py to compile performance records"]})

    def get_top_global_features_df():
        return calculate_global_shap_importance(model, model_df, top_n=15)

    render_model_insights_page(
        # model,
        model_df,
        get_model_comparison_df,
        get_top_global_features_df,
        GLOBAL_SHAP_IMAGE,
    )

elif page == "Peer Similarity Cases":
    render_similar_cases_page(
        model,
        display_df,
        model_df,
        structural_applicant_selector,
        format_single_applicant_display,
        get_similar_cases,
        get_similar_cases_by_class,
    )

elif page == "What-If Counterfactuals":
    render_counterfactual_page(
        model,
        display_df,
        model_df,
        structural_applicant_selector,
        format_single_applicant_display,
        predict_applicant,                     # 6th position
        generate_counterfactuals_for_approval, # 7th position
        render_dice_explanation_plot,          # 8th position
    )