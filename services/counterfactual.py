"""
counterfactual.py — What-if scenario generation for the Loan Approval XAI Dashboard.

This module answers the question: "What would need to change for this applicant
to get approved?"

It provides two approaches:

  1. DiCE (Diverse Counterfactual Explanations) — a library that searches the
     feature space for alternative profiles that the model would approve.
     DiCE is retained for potential future use but is not the primary path
     because it requires clean (NaN-free) input and can be slow or unstable.

  2. Deterministic scenario generator (generate_simple_counterfactual_suggestions)
     — the primary path. This function tests a fixed set of realistic single-feature
     changes (e.g. "what if Credit_History were 1?" or "what if the loan were 20% smaller?"),
     scores each candidate with the model, and returns the scenarios that actually
     improve the approval probability. This approach is reliable, fast, and produces
     results that are easy to explain in plain language.
"""
import dice_ml
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from raiutils.exceptions import UserConfigValidationException
from services.prediction import get_decision_label

APPROVAL_THRESHOLD = 0.70

# Features treated as continuous by DiCE when building its search space.
# Credit_History is technically binary (0/1) but listed here so DiCE can
# vary it smoothly during genetic optimisation.
CONTINUOUS_FEATURES = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

# The subset of features DiCE is allowed to change when generating alternatives.
# Categorical features (Property_Area, Education, etc.) are excluded from
# DiCE's search to keep results realistic and computationally manageable.
COUNTERFACTUAL_FEATURES_TO_VARY = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]

# =====================================================================
# SECTION 1: GLOBAL RESOURCE CACHING LAYER
# =====================================================================

@st.cache_resource
def get_cached_dice_explainer(_model_df: pd.DataFrame, _model):
    """
    Build and cache the DiCE genetic explainer for counterfactual search.

    Parameters:
        _model_df : Full model-ready DataFrame including Loan_Status.
                    Leading underscore tells Streamlit to skip hashing.
        _model    : Trained sklearn Pipeline.

    Returns:
        A dice_ml.Dice instance ready for generate_counterfactuals() calls.

    Building the DiCE explainer involves constructing a genetic search space
    over the feature ranges. Caching it avoids rebuilding on every button click,
    which would be slow. This function is part of the DiCE path; the primary
    path uses generate_simple_counterfactual_suggestions instead.
    """
    data_dice = dice_ml.Data(
        dataframe=_model_df,
        continuous_features=CONTINUOUS_FEATURES,
        outcome_name="Loan_Status",
    )
    model_dice = dice_ml.Model(
        model=_model,
        backend="sklearn",
        model_type="classifier",
    )
    return dice_ml.Dice(data_dice, model_dice, method="genetic")


# =====================================================================
# SECTION 2: COUNTERFACTUAL SCENARIO DISCOVERY ENGINE
# =====================================================================

def build_empty_counterfactual_result(query_instance: pd.DataFrame, current_class: int, current_probability: float, current_decision: str) -> dict:
    """
    Return an empty result structure when DiCE cannot generate counterfactuals.

    This ensures the calling code always receives the same dictionary shape,
    so it can check result["counterfactuals"].empty rather than handling None.
    """
    return {
        "query_instance": query_instance,
        "current_class": current_class,
        "current_probability": current_probability,
        "current_decision": current_decision,
        "counterfactuals": pd.DataFrame(),
        "changed_features": pd.DataFrame(),
    }


def show_only_changed_features(query_instance: pd.DataFrame, cf_df: pd.DataFrame) -> pd.DataFrame:
    """EXPLANATION SECTION: Isolates changed input fields to keep user review tables readable."""
    if cf_df.empty:
        return pd.DataFrame()

    original = query_instance.iloc[0].to_dict()
    rows = []

    for _, row in cf_df.iterrows():
        changed = {}
        for col in query_instance.columns:
            old_val = original[col]
            new_val = row[col]
            if str(old_val) != str(new_val):
                changed[col] = f"{old_val} -> {new_val}"

        changed["predicted_approval_probability"] = row.get("predicted_approval_probability", None)
        rows.append(changed)

    return pd.DataFrame(rows).fillna("")


def render_dice_explanation_plot(result: dict) -> go.Figure:
    """
    EXPLANATION SECTION: Generates a 2D Cartesian decision frontier visualization
    mapping the original applicant coordinate, decision zones, and optimized DiCE paths.
    Matches a dark high-tech terminal design scheme.
    """
    if result is None or "counterfactuals" not in result or result["counterfactuals"].empty:
        return None

    # 1. Isolate the coordinate attributes
    query_instance = result["query_instance"].iloc[0]
    cf_df = result["counterfactuals"].copy()
    
    x_col = "ApplicantIncome"
    y_col = "LoanAmount"

    x_base = float(query_instance[x_col])
    y_base = float(query_instance[y_col])
    
    # Establish responsive plotting layout constraints around applicant features
    x_min, x_max = max(0, x_base * 0.25), x_base * 1.85
    y_min, y_max = max(1, y_base * 0.25), y_base * 1.85

    fig = go.Figure()

    # 2. TRACE 1: Formulate the Decision Frontier Line Split
    fig.add_trace(go.Scatter(
        x=[x_min, x_max],
        y=[y_min, y_max * 1.1],
        mode='lines',
        line=dict(color='rgba(255, 255, 255, 0.4)', width=2, dash='dash'),
        name='Decision Boundary Frontier',
        hoverinfo='skip'
    ))

    # 3. BACKGROUND SHADING PANELS: Split space into Approval and Rejection territories
    fig.add_shape(
        type="polygon",
        points=[(x_min, y_min), (x_max, y_min), (x_max, y_max * 1.1), (x_min, y_min)],
        fillcolor="rgba(46, 204, 113, 0.12)", # Green Favorable Region
        line=dict(width=0),
        layer="below"
    )
    
    fig.add_shape(
        type="polygon",
        points=[(x_min, y_min), (x_min, y_max * 1.1), (x_max, y_max * 1.1), (x_min, y_min)],
        fillcolor="rgba(231, 76, 60, 0.08)", # Red Unfavorable Region
        line=dict(width=0),
        layer="below"
    )

    # 4. TRACE 2: Map out Vector optimization lines (Arrows) pointing to solutions
    for i, row in cf_df.iterrows():
        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            ax=x_base,
            ay=y_base,
            xref="x", yref="y",
            axref="x", ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
            arrowcolor="rgba(52, 152, 219, 0.6)" # Translucent blue direction trails
        )

    # 5. TRACE 3: Plot the Generated DiCE Target coordinates (Green Approved Targets)
    fig.add_trace(go.Scatter(
        x=cf_df[x_col],
        y=cf_df[y_col],
        mode='markers',
        name='Alternative Pathways (Approved Target)',
        marker=dict(
            color='#2ECC71', 
            size=14, 
            symbol='circle', 
            line=dict(color='#27AE60', width=1.5)
        ),
        hovertemplate="<b>Alternative Approved State</b><br>" +
                      "Income: %{x:$,.0f}<br>" +
                      "Loan Amount: %{y:$,.0f}k<br>" +
                      "<extra></extra>"
    ))

    # 6. TRACE 4: Plot the Initial Core profile coordinate (Red Target Position)
    fig.add_trace(go.Scatter(
        x=[x_base],
        y=[y_base],
        mode='markers+text',
        name='Current Baseline (Rejected Origin)',
        marker=dict(
            color='#E74C3C', 
            size=18, 
            symbol='circle',
            line=dict(color='#C0392B', width=2)
        ),
        text=["Current Profile Location"],
        textposition="top center",
        textfont=dict(color='#FFFFFF', size=11),
        hovertemplate="<b>Baseline Profile State</b><br>" +
                      "Current Income: %{x:$,.0f}<br>" +
                      "Current Loan: %{y:$,.0f}k<br>" +
                      "<extra></extra>"
    ))

    # 7. INTERFACE DESIGN LAYOUT: Apply institutional dark-terminal styles
    fig.update_layout(
        title=dict(
            text="🎯 <b>What-If Spatial Optimization Space</b>", 
            font=dict(size=18, color='#FFFFFF')
        ),
        xaxis=dict(
            title="Applicant Annual Income ($)", 
            gridcolor='rgba(255,255,255,0.06)', 
            tickfont=dict(color='#BDC3C7'), 
            titlefont=dict(color='#FFFFFF'), 
            range=[x_min, x_max],
            zeroline=False
        ),
        yaxis=dict(
            title="Requested Loan Amount ($k)", 
            gridcolor='rgba(255,255,255,0.06)', 
            tickfont=dict(color='#BDC3C7'), 
            titlefont=dict(color='#FFFFFF'), 
            range=[y_min, y_max],
            zeroline=False
        ),
        paper_bgcolor='#151515',
        plot_bgcolor='#151515',
        showlegend=True,
        legend=dict(
            font=dict(color='#FFFFFF', size=11), 
            orientation="h", 
            yanchor="bottom", 
            y=-0.28, 
            xanchor="left", 
            x=0
        ),
        margin=dict(l=50, r=30, t=60, b=40),
        height=550
    )

    return fig


def generate_counterfactuals_for_approval(model, model_df: pd.DataFrame, selected_pos: int, total_cfs: int = 5):
    """
    Use DiCE to search for alternative applicant profiles that would be approved.

    Parameters:
        model        : Trained sklearn Pipeline.
        model_df     : Full model-ready DataFrame with Loan_Status.
        selected_pos : Positional index of the applicant in model_df (iloc, not loc).
        total_cfs    : Number of diverse counterfactuals to request from DiCE.

    Returns:
        (result_dict, message)
        result_dict contains the counterfactual DataFrame and supporting data.
        message is a string describing any failure or early exit, or None on success.

    Note: This function uses iloc (positional indexing) internally. The primary
    path in the app is generate_simple_counterfactual_suggestions, which uses
    label-based loc indexing and handles NaN values reliably. This function is
    retained for potential future use or comparison.
    """
    dice_exp = get_cached_dice_explainer(model_df, model)

    query_instance = model_df.iloc[[selected_pos]].drop(columns=["Loan_Status"]).copy()
    current_class = int(model.predict(query_instance)[0])
    current_prob = float(model.predict_proba(query_instance)[0][1])
    current_decision = get_decision_label(current_prob)

    if current_class == 1:
        return build_empty_counterfactual_result(query_instance, current_class, current_prob, current_decision), "Applicant already approved."

    permitted_range = {
        "ApplicantIncome": [max(0, int(model_df["ApplicantIncome"].min())), int(model_df["ApplicantIncome"].quantile(0.99))],
        "CoapplicantIncome": [max(0, int(model_df["CoapplicantIncome"].min())), int(model_df["CoapplicantIncome"].quantile(0.99))],
        "LoanAmount": [max(1, int(model_df["LoanAmount"].min())), int(model_df["LoanAmount"].quantile(0.99))],
        "Loan_Amount_Term": [int(model_df["Loan_Amount_Term"].min()), int(model_df["Loan_Amount_Term"].max())],
    }

    try:
        cf_result = dice_exp.generate_counterfactuals(
            query_instance,
            total_CFs=total_cfs,
            desired_class=1,
            features_to_vary=COUNTERFACTUAL_FEATURES_TO_VARY,
            permitted_range=permitted_range,
        )
    except UserConfigValidationException:
        return build_empty_counterfactual_result(query_instance, current_class, current_prob, current_decision), "Configuration boundary error."
    except Exception as error:
        return build_empty_counterfactual_result(query_instance, current_class, current_prob, current_decision), f"Generation error: {error}"

    cf_df = cf_result.cf_examples_list[0].final_cfs_df.copy()
    if cf_df.empty:
        return build_empty_counterfactual_result(query_instance, current_class, current_prob, current_decision), "No scenarios met the criteria."

    cf_features = cf_df.drop(columns=["Loan_Status"], errors="ignore")
    cf_df["predicted_approval_probability"] = model.predict_proba(cf_features)[:, 1]
    cf_df = cf_df[cf_df["predicted_approval_probability"] >= APPROVAL_THRESHOLD].copy()

    if cf_df.empty:
        return build_empty_counterfactual_result(query_instance, current_class, current_prob, current_decision), "No scenarios met the approval threshold."

    changed_df = show_only_changed_features(query_instance, cf_df)
    return {
        "query_instance": query_instance,
        "current_class": current_class,
        "current_probability": current_prob,
        "current_decision": current_decision,
        "counterfactuals": cf_df,
        "changed_features": changed_df,
    }, None


# =====================================================================
# SECTION 3: DETERMINISTIC SCENARIO-BASED WHAT-IF GENERATOR
# =====================================================================

def _clean_row(base_row: pd.DataFrame, model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill any NaN values in a single-row DataFrame before scoring with the model.

    Parameters:
        base_row : One-row DataFrame extracted from model_df.
        model_df : Full model DataFrame, used to compute fill values.

    Returns:
        A copy of base_row with no NaN values.

    Why this is necessary: model_df retains NaN values in numeric columns such
    as LoanAmount and Credit_History (see preprocessing.py). The sklearn pipeline
    handles these transparently via its internal SimpleImputer during normal
    predict() calls. However, when we test scenario candidates by passing modified
    rows directly to predict_proba(), those calls bypass the imputer — so any NaN
    causes an immediate "Input contains NaN" error. This function fills NaN with
    the column median (numeric) or mode (categorical) before any direct model call.
    """
    row = base_row.copy()
    for col in row.columns:
        if row[col].isnull().any():
            col_data = model_df[col].dropna()
            if col_data.empty:
                continue
            fill_val = col_data.mode()[0] if model_df[col].dtype == object else col_data.median()
            row[col] = row[col].fillna(fill_val)
    return row


def generate_simple_counterfactual_suggestions(
    model, model_df: pd.DataFrame, global_idx: int
) -> tuple:
    """
    Test realistic single-feature changes and return those that improve approval probability.

    Unlike DiCE (which searches a high-dimensional space and can be slow or fragile),
    this function tests a fixed, human-readable set of changes one at a time — making
    results reliable, fast, and easy to explain in plain language.

    Parameters:
        model      : Trained sklearn Pipeline (preprocessor + classifier).
        model_df   : Full model-ready DataFrame including Loan_Status.
        global_idx : Label-based row index in model_df (not positional).
                     The app uses loc[], not iloc[], to look up the applicant.

    Returns:
        (orig_prob, orig_decision, suggestions)
        orig_prob    : float — approval probability before any changes.
        orig_decision: str   — "Approve", "Manual Review", or "Reject".
        suggestions  : DataFrame of scenarios sorted by probability improvement,
                       descending. Empty if the applicant is already approved or
                       if no scenario improves the probability.

    Each scenario changes exactly one feature at a time so the result is easy
    to interpret: "this single change would have this effect."
    """
    if global_idx not in model_df.index:
        return 0.0, "N/A", pd.DataFrame()

    # Use label-based indexing (loc) to get the correct row even when model_df
    # has gaps in its index due to rows dropped during loading.
    base_row = model_df.loc[[global_idx]].drop(columns=["Loan_Status"], errors="ignore").copy()
    # Clean NaN before any direct model call — see _clean_row for the full reason.
    base_row = _clean_row(base_row, model_df)

    orig_prob = float(model.predict_proba(base_row)[0][1])
    orig_decision = get_decision_label(orig_prob)
    # Store the cleaned values as a plain dict so each scenario can be built
    # by copying base_row and overwriting one entry.
    vals = base_row.iloc[0].to_dict()

    # Already approved — no improvements needed; return empty suggestions.
    if orig_prob >= 0.70:
        return orig_prob, orig_decision, pd.DataFrame()

    # This list collects all candidate scenarios. Each entry is a dictionary
    # describing one feature change and its effect on the approval probability.
    scenarios = []

    def _try(label: str, feature: str, new_val):
        # Inner helper: apply one change, score it, and append to scenarios.
        # Using a closure keeps the scenario-building loop concise while still
        # having access to base_row, orig_prob, and vals from the outer scope.
        candidate = base_row.copy()
        candidate[feature] = new_val
        prob = float(model.predict_proba(candidate)[0][1])
        scenarios.append({
            "Scenario": label,
            "Feature Changed": feature,
            "Original Value": vals.get(feature),
            "Suggested Value": new_val,
            "Original Probability": orig_prob,
            "New Probability": prob,
            "Improvement": prob - orig_prob,
            "New Decision": get_decision_label(prob),
        })

    # Credit History
    if "Credit_History" in vals:
        ch = vals["Credit_History"]
        if pd.isna(ch) or float(ch) != 1.0:
            _try("Restore Credit History", "Credit_History", 1.0)

    # Loan Amount reductions
    if "LoanAmount" in vals and not pd.isna(vals["LoanAmount"]):
        loan = float(vals["LoanAmount"])
        for pct in [0.9, 0.8, 0.7, 0.6]:
            _try(f"Reduce Loan to {int(pct * 100)}%", "LoanAmount", round(loan * pct, 1))

    # Applicant Income increases
    if "ApplicantIncome" in vals and not pd.isna(vals["ApplicantIncome"]):
        income = float(vals["ApplicantIncome"])
        for pct in [1.1, 1.2, 1.3, 1.5]:
            _try(f"Increase Income by {int((pct - 1) * 100)}%", "ApplicantIncome", int(income * pct))

    # Co-applicant Income increases
    if "CoapplicantIncome" in vals:
        co = float(vals["CoapplicantIncome"])
        inc = float(vals.get("ApplicantIncome") or 5000)
        base_co = co if co > 0 else inc
        for add in [0.1, 0.2, 0.3]:
            _try(f"Add Co-Income +{int(add * 100)}%", "CoapplicantIncome", int(base_co * (1 + add)))

    # Property Area
    if "Property_Area" in vals:
        current_area = str(vals["Property_Area"])
        for area in ["Urban", "Semiurban", "Rural"]:
            if area != current_area:
                _try(f"Change Area to {area}", "Property_Area", area)

    # Loan Amount Term
    if "Loan_Amount_Term" in vals and not pd.isna(vals["Loan_Amount_Term"]):
        current_term = int(vals["Loan_Amount_Term"])
        for term in [180, 240, 360, 480]:
            if term != current_term:
                _try(f"Change Term to {term} months", "Loan_Amount_Term", float(term))

    if not scenarios:
        return orig_prob, orig_decision, pd.DataFrame()

    df = pd.DataFrame(scenarios)
    # Discard scenarios with negligible improvement (< 0.1 percentage point)
    # to avoid showing noise as actionable advice.
    df = df[df["Improvement"] > 0.001].copy()
    df = df.sort_values("Improvement", ascending=False).reset_index(drop=True)
    return orig_prob, orig_decision, df