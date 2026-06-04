"""
similarity.py — Nearest-neighbour peer comparison for the Loan Approval XAI Dashboard.

When a user views an applicant on the Similar Cases page, this module finds
the most similar past applications in the dataset using k-nearest neighbours
(kNN) on the model's own internal feature representation.

Using the model's preprocessed feature space (rather than raw values) means
that "similar" is defined the same way the classifier defines it — applicants
that look alike to the model also appear as close neighbours here.

Caching strategy:
  - The feature transformation runs once and is cached across reruns.
  - The kNN index is built once and reused for every applicant query.
  This keeps the Similar Cases page fast even on repeated selections.
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Columns shown in the similarity results table, in display order.
# similarity_rank and similarity_distance are computed at display time;
# all other columns come directly from the original dataset.
SIMILAR_CASE_DISPLAY_COLUMNS = [
    "similarity_rank",
    "similarity_distance",
    "Loan_ID",
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
    "Loan_Status",
]


@st.cache_data
def _get_cached_transformed_features(_model, model_df: pd.DataFrame) -> np.ndarray:
    """
    Run the model's preprocessor on the full dataset and cache the result.

    Parameters:
        _model   : Trained sklearn Pipeline. The leading underscore tells
                   Streamlit not to hash this argument (sklearn objects are
                   not safely hashable).
        model_df : Full model-ready DataFrame including Loan_Status.

    Returns:
        2-D NumPy array where each row is one applicant's encoded features.

    Running the preprocessor (one-hot encoding, scaling, imputation) is
    moderately expensive. Caching means it runs once on first call and is
    reused for every subsequent similarity query in the same session.
    """
    X = model_df.drop(columns=["Loan_Status"], errors="ignore")
    preprocessor = _model.named_steps["preprocessor"]
    transformed_matrix = preprocessor.transform(X)
    # Sparse matrices from one-hot encoding must be converted to dense arrays
    # so NearestNeighbors can compute Euclidean distances row by row.
    if hasattr(transformed_matrix, "toarray"):
        return transformed_matrix.toarray()
    return np.asarray(transformed_matrix)


@st.cache_resource
def _get_cached_global_nn_index(X_transformed: np.ndarray, max_neighbors: int = 20) -> NearestNeighbors:
    """
    Build and cache a kNN index over all applicants in the dataset.

    Parameters:
        X_transformed : Encoded feature matrix (output of _get_cached_transformed_features).
        max_neighbors : Upper bound on neighbours to pre-compute.

    Returns:
        A fitted NearestNeighbors object ready for .kneighbors() queries.

    Building the kNN index involves sorting the entire feature matrix, which
    takes time proportional to the dataset size. Caching it as a resource
    means it is built once per session, regardless of how many applicants
    the user queries.
    """
    nn = NearestNeighbors(n_neighbors=min(max_neighbors, len(X_transformed)), metric="euclidean", algorithm="auto")
    return nn.fit(X_transformed)


@st.cache_resource
def _get_cached_class_nn_index(X_transformed: np.ndarray, mask_values: np.ndarray, max_neighbors: int = 20) -> tuple:
    """
    Build and cache a kNN index restricted to one outcome class (approved or rejected).

    Parameters:
        X_transformed : Full encoded feature matrix.
        mask_values   : Boolean array where True marks rows belonging to the target class.
        max_neighbors : Upper bound on neighbours to pre-compute.

    Returns:
        (fitted NearestNeighbors, class_indices)
        class_indices is a NumPy array of positions in the full matrix that belong
        to the target class. It is returned alongside the index so that query results
        (which are positions within the class subset) can be mapped back to positions
        in the original full dataset.

    A separate index per class is needed because when we search for "similar approved
    cases", we only want neighbours from the approved group — not the whole dataset.
    """
    # class_indices stores the row positions (in the full matrix) of applicants
    # that belong to the target class. This mapping is essential for translating
    # class-local neighbour positions back to positions in display_df later.
    class_indices = np.where(mask_values)[0]
    if len(class_indices) == 0:
        return None, []
    class_X = X_transformed[class_indices]
    nn = NearestNeighbors(n_neighbors=min(max_neighbors, len(class_X)), metric="euclidean", algorithm="auto")
    return nn.fit(class_X), class_indices


def _format_similar_cases_table(display_subset: pd.DataFrame, distances: list) -> pd.DataFrame:
    """
    Add rank and distance columns to a neighbour subset and reorder for display.

    Parameters:
        display_subset : Rows from display_df corresponding to the nearest neighbours.
        distances      : Euclidean distances from the query applicant to each neighbour,
                         in the same order as display_subset rows.

    Returns:
        Formatted DataFrame with only the columns in SIMILAR_CASE_DISPLAY_COLUMNS,
        plus human-readable Loan_Status labels.
    """
    result = display_subset.copy()
    result.insert(0, "similarity_distance", distances)
    result.insert(0, "similarity_rank", range(1, len(result) + 1))

    # Convert numeric Loan_Status (1/0) or string ("Y"/"N") to readable labels.
    if "Loan_Status" in result.columns:
        result["Loan_Status"] = result["Loan_Status"].map({1: "Approved", 0: "Rejected", "Y": "Approved", "N": "Rejected"})

    existing_columns = [col for col in SIMILAR_CASE_DISPLAY_COLUMNS if col in result.columns]
    return result[existing_columns]


def get_similar_cases(display_df: pd.DataFrame, model_df: pd.DataFrame, selected_pos: int, model=None, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Find the most similar applicants across the entire dataset.

    Parameters:
        display_df   : Full display DataFrame (raw values, used for the results table).
        model_df     : Full model-ready DataFrame (used to build the feature matrix).
        selected_pos : Positional index of the query applicant in the transformed matrix.
        model        : Trained sklearn Pipeline.
        n_neighbors  : Number of similar cases to return.

    Returns:
        DataFrame of the n most similar applicants, ranked by Euclidean distance
        in the model's feature space (closest first). The query applicant itself
        is excluded from the results.

    Note: selected_pos is a positional index here (not a label), because the
    transformed feature matrix is a plain NumPy array indexed from 0.
    """
    if model is None:
        raise ValueError("model must be provided to get_similar_cases().")

    X_transformed = _get_cached_transformed_features(model, model_df)
    # Request slightly more neighbours than needed so that excluding the
    # query applicant itself still leaves n valid results.
    nn_index = _get_cached_global_nn_index(X_transformed, max_neighbors=n_neighbors + 5)

    query_case = X_transformed[selected_pos : selected_pos + 1]
    distances, indices = nn_index.kneighbors(query_case)

    # Remove the query applicant from its own results (distance = 0).
    pairs = [(idx, dist) for idx, dist in zip(indices[0], distances[0]) if idx != selected_pos][:n_neighbors]
    if not pairs:
        return pd.DataFrame()

    neighbor_positions = [idx for idx, _ in pairs]
    neighbor_distances = [dist for _, dist in pairs]

    # Use iloc because neighbor_positions are positional indices in the NumPy array,
    # which correspond to positional rows in display_df (same original row order).
    subset = display_df.iloc[neighbor_positions].copy()
    return _format_similar_cases_table(subset, neighbor_distances)


def get_similar_cases_by_class(display_df: pd.DataFrame, model_df: pd.DataFrame, selected_pos: int, target_class: int, model=None, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Find the most similar applicants restricted to one outcome class.

    Parameters:
        display_df   : Full display DataFrame.
        model_df     : Full model-ready DataFrame.
        selected_pos : Positional index of the query applicant.
        target_class : 1 for approved cases, 0 for rejected cases.
        model        : Trained sklearn Pipeline.
        n_neighbors  : Number of similar cases to return.

    Returns:
        DataFrame of the n most similar applicants from the target class only,
        ranked by Euclidean distance. Used to show "Top Similar Approved Cases"
        and "Top Similar Rejected Cases" side by side.

    The kNN index for this function is built over the target class subset only,
    so distances are computed within that class — not against the full dataset.
    """
    if model is None:
        raise ValueError("model must be provided to get_similar_cases_by_class().")

    X_transformed = _get_cached_transformed_features(model, model_df)
    mask_values = (model_df["Loan_Status"] == target_class).values
    nn_index, class_indices = _get_cached_class_nn_index(X_transformed, mask_values, max_neighbors=n_neighbors + 5)

    if nn_index is None or len(class_indices) == 0:
        return pd.DataFrame()

    query_case = X_transformed[selected_pos : selected_pos + 1]
    distances, indices = nn_index.kneighbors(query_case)

    # indices[0] are positions within the class-only subset.
    # class_indices[subset_idx] maps each back to the position in the full matrix,
    # which is also the positional row in display_df (they share the same ordering).
    mapped_pairs = []
    for subset_idx, dist in zip(indices[0], distances[0]):
        original_pos = class_indices[subset_idx]
        if original_pos != selected_pos:
            mapped_pairs.append((original_pos, dist))

    mapped_pairs = mapped_pairs[:n_neighbors]
    if not mapped_pairs:
        return pd.DataFrame()

    neighbor_positions = [idx for idx, _ in mapped_pairs]
    neighbor_distances = [dist for _, dist in mapped_pairs]

    subset = display_df.iloc[neighbor_positions].copy()
    return _format_similar_cases_table(subset, neighbor_distances)
