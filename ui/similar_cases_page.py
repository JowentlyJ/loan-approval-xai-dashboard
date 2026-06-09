"""
similar_cases_page.py — Peer comparison page for the Loan Approval XAI Dashboard.

This page helps reviewers understand an applicant by comparing them against
the most similar past applications in the dataset. Similarity is measured in
the model's own feature space (after encoding and scaling), so "similar" here
means similar from the model's perspective — not just raw value proximity.

The page shows three tables:
  1. Top 10 most similar cases overall (any outcome).
  2. Top 10 most similar approved cases.
  3. Top 10 most similar rejected cases.

Tables 2 and 3 side by side let reviewers see which approved applicants this
person most resembles and which rejected ones they are close to — providing
intuition for whether the model's decision makes sense.
"""
import streamlit as st
import pandas as pd


def render_similar_cases_page(
    model,
    display_df,
    model_df,
    applicant_selector,
    format_single_applicant_display,
    get_similar_cases,
    get_similar_cases_by_class,
):
    """
    Render the Similar Cases page for the currently selected applicant.

    Parameters:
        model                       : Trained sklearn Pipeline (needed for feature transform).
        display_df                  : Full dataset with readable values for the results table.
        model_df                    : Model-ready subset for kNN feature computation.
        applicant_selector          : Callable returning the selected applicant's label index.
        format_single_applicant_display: Callable normalising applicant data to a Series.
        get_similar_cases           : Callable finding the n closest overall neighbours.
        get_similar_cases_by_class  : Callable finding the n closest neighbours of one class.

    The page layout has two levels:
      Upper: selected applicant details (left) + top 10 overall similar cases (right).
      Lower: top 10 similar approved cases (left) + top 10 similar rejected cases (right).
    """
    st.title("🔍 Similar Cases")
    st.write("Select an applicant to compare against the top 10 most similar past applications.")

    selected_pos = applicant_selector(display_df)
    
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

    # Look up the applicant's recorded loan outcome from the model-ready DataFrame.
    # Applicants excluded from model_df (missing Loan_Status) are labelled separately
    # so the UI is honest about what the dataset actually contains.
    if global_idx in model_df.index:
        current_status = model_df.loc[global_idx, "Loan_Status"]
        current_status_label = "Approved" if current_status == 1 else "Rejected"
    else:
        current_status_label = "Excluded from model data footprint"

    # ── Applicant details and top 10 overall similar cases ────────────────
    main_col1, main_col2 = st.columns(2)

    with main_col1:
        st.subheader("Selected Applicant")
        st.dataframe(
            format_single_applicant_display(applicant_display),
            use_container_width=True,
        )
        st.info(f"Observed loan status in dataset: {current_status_label}")

    with main_col2:
        # UPDATED: Changed n_neighbors metric threshold boundary limit parameter to 10
        st.subheader("Top 10 Similar Overall Cases")
        if global_idx in model_df.index:
            st.dataframe(
                get_similar_cases(
                    display_df,
                    model_df,
                    global_idx,
                    model=model,
                    n_neighbors=10,
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("Peer comparison calculations require fully valid baseline metrics profiles.")

    # ── Top 10 similar approved and rejected cases side by side ───────────
    if global_idx in model_df.index:
        st.write("---")
        col1, col2 = st.columns(2)

        with col1:
            # UPDATED: Changed n_neighbors metric threshold boundary limit parameter to 10
            st.subheader("Top 10 Similar Approved Cases")
            st.dataframe(
                get_similar_cases_by_class(
                    display_df,
                    model_df,
                    global_idx,
                    target_class=1,
                    model=model,
                    n_neighbors=10,
                ),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            # UPDATED: Changed n_neighbors metric threshold boundary limit parameter to 10
            st.subheader("Top 10 Similar Rejected Cases")
            st.dataframe(
                get_similar_cases_by_class(
                    display_df,
                    model_df,
                    global_idx,
                    target_class=0,
                    model=model,
                    n_neighbors=10,
                ),
                use_container_width=True,
                hide_index=True
            )