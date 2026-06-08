"""
loan_decision_page.py — Per-applicant prediction and SHAP explanation page.

This page is the core of the Loan Approval XAI Dashboard. For the selected applicant it shows:
  1. Applicant details table (raw display values).
  2. Prediction result — probability, progress bar, and decision label.
  3. Quick-glance KPI cards — key numbers at a glance.
  4. Why This Decision? — a SHAP bar chart with a plain-English explanation.
  5. Underwriter assessment — a manual sign-off section.

Helper functions at module level (_clean_feature_name, _explain_shap_chart)
handle display formatting and explanation text generation so the render function
stays focused on layout.
"""
import re
import pandas as pd
import streamlit as st


def _clean_feature_name(raw: str) -> str:
    """Convert a sklearn ColumnTransformer feature name to plain English."""
    name = re.sub(r"^(num__|cat__)", "", raw)
    # One-hot encoded categoricals: split on last underscore group
    # e.g. Property_Area_Semiurban → "Property area: Semiurban"
    #      Married_Yes             → "Married: Yes"
    #      Dependents_1.0          → "Dependents: 1"
    _known_bases = [
        "Property_Area", "Self_Employed", "Education",
        "Married", "Gender", "Dependents",
    ]
    for base in _known_bases:
        if name.startswith(base + "_"):
            suffix = name[len(base) + 1:]
            # Clean up float-style values like "1.0" → "1"
            try:
                suffix = str(int(float(suffix)))
            except ValueError:
                pass
            label = base.replace("_", " ").title()
            return f"{label}: {suffix}"
    # Plain numeric features
    _plain = {
        "Credit_History":     "Credit history",
        "LoanAmount":         "Loan amount",
        "ApplicantIncome":    "Applicant income",
        "CoapplicantIncome":  "Co-applicant income",
        "Loan_Amount_Term":   "Loan term",
    }
    if name in _plain:
        return _plain[name]
    return name.replace("_", " ").title()


def _explain_shap_chart(
    explanation_df: pd.DataFrame, decision: str, probability: float
) -> str:
    """
    Generate a 3–5 sentence plain-English explanation of a local SHAP bar chart.

    Parameters:
        explanation_df : DataFrame from get_local_shap_values (columns: feature, shap_value).
        decision       : The model's decision label for this applicant.
        probability    : The approval probability (0–1 float).

    Returns:
        A markdown-formatted string ready for st.markdown(). Returns "" if
        explanation_df is empty (so callers can check before rendering).

    The explanation identifies the single strongest positive and negative feature,
    characterises whether positive or negative forces dominated overall, and states
    the final outcome in plain language — without referring to log-odds or SHAP
    terminology that a non-technical reader would not understand.
    """
    if explanation_df.empty:
        return ""

    positives = explanation_df[explanation_df["shap_value"] > 0]
    negatives = explanation_df[explanation_df["shap_value"] < 0]

    total_pos = positives["shap_value"].sum()
    total_neg = negatives["shap_value"].sum()   # negative number

    top_pos = (
        _clean_feature_name(positives.loc[positives["shap_value"].idxmax(), "feature"])
        if not positives.empty else None
    )
    top_neg = (
        _clean_feature_name(negatives.loc[negatives["shap_value"].idxmin(), "feature"])
        if not negatives.empty else None
    )

    intro = (
        "This chart shows which profile details influenced the model's decision. "
        "**Green bars** represent features that increased the chance of approval; "
        "**red bars** represent features that worked against it. "
        "Longer bars indicate stronger influence. The SHAP value measures "
        "direction and strength of each feature's push, not the final probability directly."
    )

    if top_pos and top_neg:
        factors = (
            f"**{top_neg}** had the strongest negative effect, lowering the model's "
            f"confidence in approval. **{top_pos}** had the strongest positive effect, "
            f"helping the application."
        )
    elif top_neg:
        factors = (
            f"**{top_neg}** had the strongest negative effect, lowering the model's "
            f"confidence in approval. No features clearly pushed toward approval."
        )
    elif top_pos:
        factors = (
            f"**{top_pos}** had the strongest positive effect, helping the application. "
            f"No features clearly pushed against approval."
        )
    else:
        factors = "The feature influences were mixed or near-zero."

    if abs(total_neg) > total_pos * 1.5:
        balance = "The negative influences were considerably stronger overall."
    elif total_pos > abs(total_neg) * 1.5:
        balance = "The positive influences were considerably stronger overall."
    else:
        balance = "The positive and negative influences were roughly balanced."

    outcome = (
        f"Together, these factors produced a **{decision}** decision "
        f"with an approval probability of **{probability:.1%}**."
    )

    return f"{intro}\n\n{factors} {balance} {outcome}"


def render_loan_decision_page(
    model,
    display_df,
    model_df,
    applicant_selector,
    format_single_applicant_display,
    predict_applicant,
    create_selected_application_kpi_chart,   # None — deprecated chart, kept for signature compat
    create_selected_application_decision_chart,  # None — deprecated
    create_selected_application_binary_chart,    # None — deprecated
    get_local_shap_values,
    create_local_shap_bar_chart,
    render_underwriter_assessment_section,
):
    """
    Render the Loan Decision page for the currently selected applicant.

    Parameters:
        model                               : Trained sklearn Pipeline.
        display_df                          : Full dataset with readable values.
        model_df                            : Model-ready subset for prediction/SHAP.
        applicant_selector                  : Callable returning the selected applicant's
                                              original label index.
        format_single_applicant_display     : Callable normalising applicant data to a Series.
        predict_applicant                   : Callable running the model on a DataFrame row.
        create_selected_application_*       : Deprecated chart functions (pass None).
        get_local_shap_values               : Callable computing per-applicant SHAP values.
        create_local_shap_bar_chart         : Callable building the SHAP Plotly figure.
        render_underwriter_assessment_section: Callable rendering the manual sign-off form.
    """
    st.title("⚖️ Loan Decision")
    st.write("Select a loan application to view applicant details and the model decision.")

    selected_pos = applicant_selector(display_df)

    # Resolve selected_pos to a label index (global_idx) and a display row.
    # selected_pos is an integer label from the original RangeIndex, which is
    # valid for both display_df.iloc[] and model_df.loc[].
    if isinstance(selected_pos, str):
        clean_id = selected_pos.split(" ")[0].strip()
        matching_rows = display_df[display_df["Loan_ID"] == clean_id]
        if matching_rows.empty:
            st.error(f"⚠️ Sync Failure: Chosen Application ID '{clean_id}' could not be resolved.")
            return
        applicant_display = matching_rows.iloc[0]
        global_idx = matching_rows.index[0]
    else:
        global_idx = int(selected_pos)
        applicant_display = display_df.iloc[global_idx]

    # Default values used if the applicant is absent from model_df (missing Loan_Status).
    # The right column checks model_df.index before overwriting these, so they act
    # as safe fallbacks for the KPI card section which runs after the columns.
    probability = 0.0
    decision = "N/A"
    prediction = 0

    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        st.subheader("Applicant Details")
        st.dataframe(
            format_single_applicant_display(applicant_display),
            use_container_width=True,
        )

    with right_col:
        st.subheader("Prediction Result")
        if global_idx in model_df.index:
            applicant_model_row = model_df.loc[[global_idx]].drop(columns=["Loan_Status"], errors="ignore")
            prediction, probability, decision = predict_applicant(model, applicant_model_row)

            st.metric("Approval Probability", f"{probability:.2%}")
            st.progress(int(probability * 100))
            st.metric("Predicted Class", "Approved (1)" if prediction == 1 else "Rejected (0)")
            st.metric("Decision Label", decision)

            if decision == "Approve":
                st.success("Recommended decision: Approve")
            elif decision == "Manual Review":
                st.warning("Recommended decision: Manual Review")
            else:
                st.error("Recommended decision: Reject")
        else:
            st.error("❌ Real-Time Inference Unavailable")
            st.info("ℹ️ Profile dropped from training dataset due to missing values.")

    # ── Quick-glance KPI cards ─────────────────────────────────────────────
    # Six side-by-side metric cards give a fast overview without requiring the
    # user to read the full applicant table. Values come from applicant_display
    # (readable) and from the prediction result computed above (probability/decision).
    # LoanAmount is stored in thousands in the dataset, so it is shown with a "K" suffix.
    # Loan/Income ratio is computed here rather than stored; it is None when either
    # value is missing or zero to avoid division errors.
    if global_idx in model_df.index:
        st.divider()
        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        with kpi1:
            st.metric("Approval Prob.", f"{probability:.1%}")
        with kpi2:
            st.metric("Decision", decision)
        with kpi3:
            income_val = applicant_display.get("ApplicantIncome", 0)
            st.metric("Income", f"${int(income_val or 0):,}")
        with kpi4:
            loan_raw = applicant_display.get("LoanAmount", None)
            if loan_raw is not None and not pd.isna(loan_raw):
                st.metric("Loan Amount", f"{int(loan_raw)}K")
            else:
                st.metric("Loan Amount", "N/A")
        with kpi5:
            ch = applicant_display.get("Credit_History", None)
            if ch is None or pd.isna(ch):
                ch_label = "N/A"
            elif ch == 1:
                ch_label = "Yes"
            else:
                ch_label = "No"
            st.metric("Credit History", ch_label)
        with kpi6:
            inc = int(applicant_display.get("ApplicantIncome", 0) or 0)
            loa_raw = applicant_display.get("LoanAmount", None)
            loa = (float(loa_raw) * 1000) if (loa_raw is not None and not pd.isna(loa_raw)) else 0
            lti = round(loa / inc, 2) if inc > 0 and loa > 0 else None
            st.metric("Loan/Income", f"{lti:.2f}x" if lti is not None else "N/A")

    # ── Deprecated chart block ─────────────────────────────────────────────
    # These three charts were replaced by the KPI cards and SHAP section above.
    # The guard on create_selected_application_kpi_chart is not None means this
    # block is permanently skipped (None is passed from app.py), so it has no
    # effect on the running app but is retained to avoid breaking the function signature.
    if global_idx in model_df.index and create_selected_application_kpi_chart is not None:
        st.divider()
        st.subheader("Selected Application Charts")
        chart_col1, chart_col2 = st.columns(2)

        applicant_model_row = model_df.loc[[global_idx]].drop(columns=["Loan_Status"], errors="ignore")

        with chart_col1:
            try:
                st.plotly_chart(
                    create_selected_application_kpi_chart(applicant_model_row.iloc[0]),
                    use_container_width=True,
                )
            except Exception as chart_err:
                st.caption(f"KPI Chart bypassed: {chart_err}")

        with chart_col2:
            try:
                st.plotly_chart(
                    create_selected_application_decision_chart(probability),
                    use_container_width=True,
                )
            except Exception as chart_err:
                st.caption(f"Decision Chart bypassed: {chart_err}")

        try:
            st.plotly_chart(
                create_selected_application_binary_chart(applicant_model_row.iloc[0], prediction),
                use_container_width=True,
            )
        except Exception as chart_err:
            pass

    # ── SHAP explanation section ───────────────────────────────────────────
    # This section is independent of the deprecated chart guard above so it
    # always renders for any applicant that exists in model_df, regardless of
    # whether the old chart functions are present.
    if global_idx in model_df.index:
        st.divider()
        st.subheader("Why This Decision? (XAI)")

        try:
            explanation_df = get_local_shap_values(model, model_df, global_idx)
            fig = create_local_shap_bar_chart(explanation_df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                shap_text = _explain_shap_chart(explanation_df, decision, probability)
                if shap_text:
                    with st.expander("How to read this chart", expanded=True):
                        st.markdown(shap_text)
            else:
                st.dataframe(explanation_df, use_container_width=True, hide_index=True)
        except Exception as error:
            st.warning("The SHAP explanation could not be generated for this applicant.")
            st.caption(f"Technical detail: {error}")

    st.divider()
    st.subheader("📝 Underwriter Assessment & Manual Sign-Off")
    
    if render_underwriter_assessment_section is not None:
        # FIXED: Pass the required object matrix down to satisfy line 58 of assessment.py
        render_underwriter_assessment_section(applicant_display, probability, decision)