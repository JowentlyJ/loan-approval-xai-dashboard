"""
counterfactual_page.py — What-if scenario display for the Loan Approval XAI Dashboard.

This page answers the question "what could this applicant change to get approved?"
It uses the deterministic scenario generator from services/counterfactual.py to
test realistic single-feature changes, then displays:
  1. A table of all improving scenarios with probabilities and decision labels.
  2. A horizontal bar chart of probability improvement per scenario.
  3. A plain-English summary of the strongest improvement.
  4. Expandable explanation cards for each scenario with realism labels.

For approved applicants the page shows a friendly info message and stops early.
For applicants with missing values the page shows an error and stops early.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from services.counterfactual import generate_simple_counterfactual_suggestions

# ── Realism tiers ──────────────────────────────────────────────────────────
# Maps the feature that was changed to a plain-language label describing how
# easy it is for a real applicant to make that change. These labels appear in
# the scenario expander headers so reviewers can quickly judge actionability.
_REALISM = {
    "Property_Area":       "Less actionable",
    "Loan_Amount_Term":    "Potentially actionable",
    "LoanAmount":          "More actionable",
    "Credit_History":      "More actionable, but takes time",
    "ApplicantIncome":     "Potentially actionable",
    "CoapplicantIncome":   "Potentially actionable",
}


def _fmt(feature: str, val) -> str:
    """Format a feature value for display in plain-English text."""
    if val is None:
        return "N/A"
    try:
        if feature == "LoanAmount":
            return f"{float(val):.0f}K"
        if feature in ("ApplicantIncome", "CoapplicantIncome"):
            return f"${int(float(val)):,}"
        if feature == "Loan_Amount_Term":
            return f"{int(float(val))} months"
        if feature == "Credit_History":
            return f"{int(float(val))}"
    except (ValueError, TypeError):
        pass
    return str(val)


def _explain_scenario(row: "pd.Series", orig_prob: float) -> dict:
    """
    Build a plain-English explanation for one what-if scenario row.

    Parameters:
        row       : One row from the suggestions DataFrame (output of
                    generate_simple_counterfactual_suggestions), containing
                    Feature Changed, Original Value, Suggested Value,
                    New Probability, Improvement, and New Decision.
        orig_prob : The applicant's original approval probability (0–1).

    Returns:
        Dictionary with keys:
          "text"        : Markdown explanation string (3–4 sentences).
          "realism"     : Human-readable actionability label from _REALISM.
          "new_decision": "Approve", "Manual Review", or "Reject".

    Feature-specific branches write different sentences depending on which
    feature changed. A review_note is appended when the new decision is only
    "Manual Review", alerting the reader that the improvement is not sufficient
    for automatic approval.
    """
    feature    = row["Feature Changed"]
    orig_val   = row["Original Value"]
    new_val    = row["Suggested Value"]
    new_prob   = row["New Probability"]
    improvement_pp = row["Improvement"] * 100
    new_decision   = row["New Decision"]

    realism = _REALISM.get(feature, "Potentially actionable")
    is_review = new_decision == "Manual Review"

    review_note = (
        " This improvement alone is not enough for automatic approval — "
        "additional supporting factors may be needed to reach a full Approve decision."
        if is_review else ""
    )

    if feature == "Property_Area":
        text = (
            f"Changing the property area from **{orig_val}** to **{new_val}** "
            f"raises the approval probability from {orig_prob:.1%} to {new_prob:.1%} "
            f"(+{improvement_pp:.1f} pp), resulting in a **{new_decision}** decision. "
            f"The model treats {new_val} areas more favourably than {orig_val}. "
            f"However, property location is usually fixed once a home has been selected, "
            f"so this scenario reveals model sensitivity to location rather than an easy action the applicant can take."
        )

    elif feature == "Loan_Amount_Term":
        direction = "shorter" if float(new_val) < float(orig_val) else "longer"
        text = (
            f"Changing the loan term from **{_fmt(feature, orig_val)}** to **{_fmt(feature, new_val)}** "
            f"raises the approval probability from {orig_prob:.1%} to {new_prob:.1%} "
            f"(+{improvement_pp:.1f} pp), resulting in a **{new_decision}** decision. "
            f"A {direction} term can sometimes be negotiated with the lender. "
            f"Note that a shorter term raises monthly repayments, so the applicant's "
            f"affordability should be verified before pursuing this option."
        )

    elif feature == "LoanAmount":
        strength = (
            "relatively small and may not be sufficient on its own"
            if improvement_pp < 5 else
            "meaningful and could make the application viable"
        )
        text = (
            f"Reducing the loan amount from **{_fmt(feature, orig_val)}** to **{_fmt(feature, new_val)}** "
            f"raises the approval probability from {orig_prob:.1%} to {new_prob:.1%} "
            f"(+{improvement_pp:.1f} pp), resulting in a **{new_decision}** decision. "
            f"This can be achieved by increasing the down payment, choosing a less expensive property, "
            f"or simply requesting less financing. The improvement is {strength}."
        )

    elif feature == "Credit_History":
        text = (
            f"Restoring credit history from **{_fmt(feature, orig_val)}** to **{_fmt(feature, new_val)}** "
            f"raises the approval probability from {orig_prob:.1%} to {new_prob:.1%} "
            f"(+{improvement_pp:.1f} pp), resulting in a **{new_decision}** decision. "
            f"This is typically the strongest single change available. "
            f"Improving credit history takes time — it usually requires resolving outstanding "
            f"debts and maintaining a clean repayment record — but it has by far the greatest impact on approval."
        )

    elif feature == "ApplicantIncome":
        text = (
            f"Increasing the applicant's income from **{_fmt(feature, orig_val)}** to **{_fmt(feature, new_val)}** "
            f"raises the approval probability from {orig_prob:.1%} to {new_prob:.1%} "
            f"(+{improvement_pp:.1f} pp), resulting in a **{new_decision}** decision. "
            f"While income cannot always be changed immediately, demonstrating additional income "
            f"sources, a salary increase, or a secondary job could support this."
        )

    elif feature == "CoapplicantIncome":
        text = (
            f"Increasing co-applicant income from **{_fmt(feature, orig_val)}** to **{_fmt(feature, new_val)}** "
            f"raises the approval probability from {orig_prob:.1%} to {new_prob:.1%} "
            f"(+{improvement_pp:.1f} pp), resulting in a **{new_decision}** decision. "
            f"Adding a co-applicant or demonstrating higher combined household income could achieve this."
        )

    else:
        text = (
            f"Changing **{feature}** from **{orig_val}** to **{new_val}** "
            f"raises the approval probability from {orig_prob:.1%} to {new_prob:.1%} "
            f"(+{improvement_pp:.1f} pp), resulting in a **{new_decision}** decision."
        )

    return {"text": text + review_note, "realism": realism, "new_decision": new_decision}


def render_counterfactual_page(
    model,
    display_df,
    model_df,
    applicant_selector,
    format_single_applicant_display,
    predict_applicant,
    generate_counterfactuals_for_approval,  # DiCE path — retained for signature compat, not called
    render_dice_explanation_plot,           # DiCE plot — retained for signature compat, not called
):
    """
    Render the What-If Counterfactuals page for the selected applicant.

    The page only generates scenarios on button click (not on every rerun).
    This avoids running predict_proba dozens of times every time the user
    adjusts a filter or changes navigation.

    The primary scenario generator is generate_simple_counterfactual_suggestions,
    imported directly at the top of this file. The DiCE parameters are retained
    in the signature so app.py does not need to change if DiCE is re-enabled later.
    """
    st.title("🔁 Counterfactuals")
    st.write(
        "Explore which single-feature changes would improve the model's approval decision "
        "for the selected applicant."
    )

    selected_pos = applicant_selector(display_df)

    if isinstance(selected_pos, str):
        clean_id = selected_pos.split(" ")[0].strip()
        matching_rows = display_df[display_df["Loan_ID"] == clean_id]
        if matching_rows.empty:
            st.error(f"⚠️ Application ID '{clean_id}' could not be located.")
            return
        applicant_display = matching_rows.iloc[0]
        global_idx = matching_rows.index[0]
    else:
        global_idx = int(selected_pos)
        applicant_display = display_df.iloc[global_idx]

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.subheader("Selected Applicant")
        formatted = format_single_applicant_display(applicant_display)
        if isinstance(formatted, pd.Series):
            dm = pd.DataFrame(formatted)
            dm.columns = ["Value"]
        elif isinstance(formatted, pd.DataFrame):
            dm = formatted.T
            dm.columns = ["Value"]
        else:
            dm = pd.DataFrame(applicant_display)
            dm.columns = ["Value"]
        st.dataframe(dm, use_container_width=True)

    with right_col:
        if global_idx not in model_df.index:
            st.subheader("Current Prediction")
            st.error("❌ Applicant has missing values — counterfactuals unavailable.")
            return

        applicant_model_row = model_df.loc[[global_idx]].drop(columns=["Loan_Status"], errors="ignore")
        _, probability, decision = predict_applicant(model, applicant_model_row)

        st.subheader("Current Prediction")
        st.metric("Approval Probability", f"{probability:.2%}")
        st.progress(int(probability * 100))
        st.metric("Decision", decision)

        if decision == "Approve":
            st.success("This applicant is already approved. No what-if changes are needed.")
            return

        if decision == "Manual Review":
            st.info("In Manual Review — showing scenarios that could reach full Approval.")

        run_scenarios = st.button("Generate What-If Scenarios", use_container_width=True)

    if not run_scenarios:
        return

    st.divider()

    with st.spinner("Scoring what-if scenarios…"):
        orig_prob, _, suggestions = generate_simple_counterfactual_suggestions(
            model, model_df, global_idx
        )

    if suggestions.empty:
        st.info(
            "No single-feature change was found that improves the approval probability "
            "for this applicant. Multiple simultaneous changes may be required."
        )
        return

    # ── Scenario table ─────────────────────────────────────────────────────
    # tbl is a formatted copy of suggestions — probabilities shown as percentages
    # and improvement prefixed with "+" so the table is self-explanatory.
    # The original suggestions DataFrame keeps raw floats so downstream code
    # (bar chart, explanation loop) can still do arithmetic on the values.
    st.subheader("What-If Scenarios")
    display_cols = [
        "Scenario", "Feature Changed", "Original Value", "Suggested Value",
        "Original Probability", "New Probability", "Improvement", "New Decision",
    ]
    tbl = suggestions[display_cols].copy()
    tbl["Original Probability"] = tbl["Original Probability"].map("{:.1%}".format)
    tbl["New Probability"] = tbl["New Probability"].map("{:.1%}".format)
    tbl["Improvement"] = tbl["Improvement"].map("+{:.1%}".format)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── Probability improvement bar chart ──────────────────────────────────
    # Bar colour signals impact at a glance: green (>25 pp) is a strong change,
    # amber (10–25 pp) is moderate, and blue (<10 pp) is a small improvement.
    st.subheader("Probability Improvement by Scenario")
    top = suggestions.head(10)
    colours = [
        "#2ECC71" if v >= 0.25 else ("#F39C12" if v >= 0.10 else "#3498DB")
        for v in top["Improvement"]
    ]
    fig = go.Figure(go.Bar(
        x=top["Improvement"] * 100,
        y=top["Scenario"],
        orientation="h",
        marker_color=colours,
        text=[f"+{v:.1%}" for v in top["Improvement"]],
        textposition="outside",
        hovertemplate="%{y}<br>Improvement: +%{x:.1f} pp<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Approval Probability Improvement (pp)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=70, t=20, b=40),
        height=max(280, len(top) * 38 + 60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Plain-English summary ──────────────────────────────────────────────
    st.subheader("Summary")
    best = suggestions.iloc[0]
    st.info(
        f"**Strongest improvement:** *{best['Scenario']}* — "
        f"changing **{best['Feature Changed']}** from `{best['Original Value']}` "
        f"to `{best['Suggested Value']}` raises the approval probability from "
        f"**{orig_prob:.1%}** to **{best['New Probability']:.1%}** "
        f"(+{best['Improvement']:.1%}), resulting in a **{best['New Decision']}** decision."
    )

    approve_count = (suggestions["New Decision"] == "Approve").sum()
    review_count = (suggestions["New Decision"] == "Manual Review").sum()
    if approve_count > 0:
        st.success(f"{approve_count} scenario(s) would result in automatic **Approval**.")
    if review_count > 0:
        st.warning(f"{review_count} scenario(s) would move the decision to **Manual Review**.")

    # ── Scenario Explanations ──────────────────────────────────────────────
    # One collapsible expander per scenario keeps the page scannable while
    # still offering full detail on demand. The expander label includes the
    # decision icon and realism label so users can filter visually without
    # opening every card.
    st.subheader("Scenario Explanations")
    st.caption("Expand each scenario for a plain-language explanation and practicality note.")

    _DECISION_ICON = {"Approve": "✅", "Manual Review": "🔶", "Reject": "❌"}
    _REALISM_ICON  = {
        "Less actionable":               "🔴",
        "Potentially actionable":        "🟡",
        "More actionable":               "🟢",
        "More actionable, but takes time": "🟢",
    }

    for i, (_, row) in enumerate(suggestions.iterrows()):
        expl = _explain_scenario(row, orig_prob)
        decision_icon = _DECISION_ICON.get(expl["new_decision"], "")
        realism_icon  = _REALISM_ICON.get(expl["realism"], "🟡")
        label = (
            f"{i + 1}. {row['Scenario']}  "
            f"{decision_icon} {expl['new_decision']}  ·  "
            f"{realism_icon} {expl['realism']}"
        )
        with st.expander(label, expanded=False):
            m1, m2, m3 = st.columns(3)
            m1.metric("Original Probability", f"{orig_prob:.1%}")
            m2.metric("New Probability",       f"{row['New Probability']:.1%}")
            m3.metric("Improvement",           f"+{row['Improvement'] * 100:.1f} pp")
            st.markdown(expl["text"])
