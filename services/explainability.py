"""
explainability.py — SHAP-based model explanation for the Loan Approval XAI Dashboard.

This module answers the question "why did the model make this decision?"
at two levels:

  Local (per-applicant): Which features pushed this specific prediction toward
  approval or rejection, and by how much? (get_local_shap_values)

  Global (dataset-wide): Which features are most influential overall, averaged
  across all applicants? (calculate_global_shap_importance)

SHAP (SHapley Additive exPlanations) assigns each feature a value that
represents its contribution to moving the model's output away from the
average. Positive values push toward approval; negative values push against it.
"""
import os
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
import streamlit as st

# =====================================================================
# SECTION 1: LOCAL SINGLE-ROW SHAP ATTRIBUTION EXPLAINER
# =====================================================================

def get_local_shap_values(_model, model_df: pd.DataFrame, selected_pos: int) -> pd.DataFrame:
    """
    Calculate SHAP values for a single applicant to explain their prediction.

    Parameters:
        _model      : Trained sklearn Pipeline. Leading underscore prevents
                      Streamlit from trying to hash the complex sklearn object.
        model_df    : Full model-ready DataFrame including Loan_Status.
        selected_pos: Label-based index of the applicant in model_df (not
                      a positional row number — use model_df.loc, not .iloc).

    Returns:
        DataFrame with columns "feature" and "shap_value", sorted by the
        absolute SHAP value descending (most influential features first).
        Positive shap_value → feature pushed toward approval.
        Negative shap_value → feature pushed toward rejection.

    This function chooses between TreeExplainer (for tree-based models such as
    Random Forest or Gradient Boosting) and LinearExplainer (for logistic
    regression) based on the classifier's class name. Both explainers operate
    on the model's transformed feature space, not the raw input columns.
    """
    X = model_df.drop(columns=["Loan_Status"], errors="ignore")

    preprocessor = _model.named_steps["preprocessor"]
    classifier = _model.named_steps["classifier"]

    # Transform the full dataset matrix to act as background reference data for linear models
    X_all_transformed = preprocessor.transform(X)
    if hasattr(X_all_transformed, "toarray"):
        X_all_transformed = X_all_transformed.toarray()
    else:
        X_all_transformed = np.asarray(X_all_transformed)

    # model_df may have gaps in its index because rows with missing Loan_Status
    # were dropped during loading. get_loc() converts the label (e.g. 42) to
    # its actual position in the transformed NumPy array (e.g. row 38), so SHAP
    # reads the correct applicant's features rather than a different row.
    if selected_pos in X.index:
        row_idx = X.index.get_loc(selected_pos)
    else:
        row_idx = 0

    single_row_transformed = X_all_transformed[row_idx : row_idx + 1]

    # Dynamic Explainer Selector for Local Single-Row Inferences
    cls_name = type(classifier).__name__
    if "Logistic" in cls_name or "Linear" in cls_name:
        explainer = shap.LinearExplainer(classifier, X_all_transformed)
        raw_shap_values = explainer.shap_values(single_row_transformed)
    else:
        explainer = shap.TreeExplainer(classifier)
        raw_shap_values = explainer.shap_values(single_row_transformed)

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = X.columns.tolist()

    # SHAP output shape varies by explainer and sklearn version:
    #   list form: [shap_class_0, shap_class_1] — take index [1] for approval probability.
    #   3-D array: shape (n_samples, n_features, n_classes) — slice class 1.
    #   2-D array: shape (n_samples, n_features) — already the values we need.
    if isinstance(raw_shap_values, list):
        shap_values = raw_shap_values[1] if len(raw_shap_values) > 1 else raw_shap_values[0]
    elif len(raw_shap_values.shape) == 3:
        shap_values = raw_shap_values[:, :, 1]
    else:
        shap_values = raw_shap_values

    applicant_values = shap_values[0]
    # Guard against shape mismatches between the preprocessor's feature names
    # and the SHAP values array length (can occur with some sklearn versions).
    min_length = min(len(feature_names), len(applicant_values))

    return pd.DataFrame({
        "feature": feature_names[:min_length],
        "shap_value": applicant_values[:min_length]
    }).sort_values(by="shap_value", key=abs, ascending=False)


# =====================================================================
# SECTION 2: INTERACTIVE LOCAL SHAP CHART BUILDER
# =====================================================================

def create_local_shap_bar_chart(explanation_df: pd.DataFrame, max_features: int = 10):
    """
    Build a horizontal Plotly bar chart showing each feature's SHAP contribution.

    Parameters:
        explanation_df : DataFrame from get_local_shap_values (columns: feature, shap_value).
        max_features   : Maximum number of features to display (top by absolute value).

    Returns:
        A Plotly Figure object. Returns an empty Figure if explanation_df is empty.

    Green bars represent positive SHAP values (feature helped the application).
    Red bars represent negative SHAP values (feature worked against approval).
    The chart is rendered on the Loan Decision page below the prediction result.
    """
    if explanation_df.empty:
        return go.Figure()

    plot_df = explanation_df.head(max_features).copy().sort_values(by="shap_value", ascending=True)
    colors = ["#2ecc71" if val >= 0 else "#e74c3c" for val in plot_df["shap_value"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_df["feature"],
        x=plot_df["shap_value"],
        orientation='h',
        marker_color=colors,
        text=plot_df["shap_value"].round(4),
        textposition='auto'
    ))

    fig.update_layout(
        title="Local Feature Contributions",
        xaxis_title="Contribution Score (direction and strength of influence)",
        yaxis_title="Applicant Profile Feature",
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        template="plotly_white"
    )

    return fig


# =====================================================================
# SECTION 3: GLOBAL SHAP PLOTTING INTERFACE UTILITY
# =====================================================================

def plot_global_feature_importance(global_importance_df: pd.DataFrame, top_n: int = 10):
    """
    Generates an aggregated macro overview feature rank chart using Plotly.
    """
    if global_importance_df.empty:
        return go.Figure()

    plot_df = global_importance_df.head(top_n).copy().sort_values(by="mean_abs_shap", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_df["feature"],
        x=plot_df["mean_abs_shap"],
        orientation='h',
        marker_color="#34495e",
        text=plot_df["mean_abs_shap"].round(4),
        textposition='auto'
    ))

    fig.update_layout(
        title="Overall Feature Importance (Average Impact Score)",
        xaxis_title="Average Impact Score",
        yaxis_title="Feature Name",
        margin=dict(l=20, r=20, t=40, b=20),
        height=450,
        template="plotly_white"
    )

    return fig


# =====================================================================
# SECTION 4: ADAPTIVE GLOBAL SHAP DATA COMPUTATION & CACHING
# =====================================================================

@st.cache_data
def _compute_cached_global_shap_data(_model, model_df: pd.DataFrame):
    """
    Compute SHAP values for every applicant in the dataset and cache the result.

    Parameters:
        _model   : Trained sklearn Pipeline (leading underscore skips hashing).
        model_df : Full model-ready DataFrame.

    Returns:
        (shap_values, X_transformed, feature_names)
        shap_values    : 2-D array of shape (n_samples, n_features).
        X_transformed  : Encoded feature matrix.
        feature_names  : List of feature names from the preprocessor.

    Computing SHAP values for the entire dataset is expensive, so the result
    is cached. This function feeds both the global importance table and any
    global SHAP summary plots generated during model evaluation.
    """
    X = model_df.drop(columns=["Loan_Status"], errors="ignore")
    
    preprocessor = _model.named_steps["preprocessor"]
    classifier = _model.named_steps["classifier"]

    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    else:
        X_transformed = np.asarray(X_transformed)

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = X.columns.tolist()

    cls_name = type(classifier).__name__
    if "Logistic" in cls_name or "Linear" in cls_name:
        explainer = shap.LinearExplainer(classifier, X_transformed)
        raw_shap_values = explainer.shap_values(X_transformed)
    else:
        explainer = shap.TreeExplainer(classifier)
        raw_shap_values = explainer.shap_values(X_transformed)

    if isinstance(raw_shap_values, list):
        shap_values = raw_shap_values[1] if len(raw_shap_values) > 1 else raw_shap_values[0]
    elif len(raw_shap_values.shape) == 3:
        shap_values = raw_shap_values[:, :, 1]
    else:
        shap_values = raw_shap_values

    return shap_values, X_transformed, feature_names


def calculate_global_shap_importance(model, model_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """
    Compute which features are most influential across the entire dataset.

    Parameters:
        model    : Trained sklearn Pipeline.
        model_df : Full model-ready DataFrame.
        top_n    : Number of top features to return.

    Returns:
        DataFrame with columns "feature" and "mean_abs_shap", sorted descending.
        mean_abs_shap is the average of |SHAP value| across all applicants,
        which measures how much each feature typically moves the model's output
        regardless of direction (positive or negative).

    This gives a dataset-level view of feature importance, complementing the
    per-applicant local explanations shown on the Loan Decision page.
    """
    try:
        shap_values, _, feature_names = _compute_cached_global_shap_data(model, model_df)
        
        # Take mean absolute values across all instances
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        min_length = min(len(feature_names), len(mean_abs_shap))
        
        df_importance = pd.DataFrame({
            "feature": feature_names[:min_length],
            "mean_abs_shap": mean_abs_shap[:min_length]
        }).sort_values(by="mean_abs_shap", ascending=False)
        
        return df_importance.head(top_n)
    except Exception as e:
        print(f"🚨 GLOBAL SHAP ERROR TRACE: {str(e)}")
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])