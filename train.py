import os
import time
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from config import MODEL_DIR, OUTPUT_DIR, BEST_MODEL_PATH
from preprocessing import load_clean_data, split_features_target, get_feature_columns

# =====================================================================
# SECTION 1: ADVANCED REGULATORY FAIRNESS AUDITING
# =====================================================================

def audit_fairness_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive_feature: pd.Series) -> dict:
    """
    EXPLANATION SECTION: Computes compliance-standard risk adjustments.
    Tracks Selection Rates, Disparate Impact (the 4/5ths rule), and 
    Equal Opportunity metrics across sensitive cohorts (e.g., Gender).
    """
    cohorts = sensitive_feature.dropna().unique()
    selection_rates = {}
    true_positive_rates = {}

    for cohort in cohorts:
        mask = (sensitive_feature == cohort)
        if mask.sum() == 0:
            continue
        
        selection_rates[cohort] = float(y_pred[mask].mean())
        
        actual_positives = (y_true == 1) & mask
        if actual_positives.sum() > 0:
            true_positive_rates[cohort] = float(y_pred[actual_positives].mean())
        else:
            true_positive_rates[cohort] = 0.0

    sorted_rates = sorted(selection_rates.values())
    demographic_parity_diff = sorted_rates[-1] - sorted_rates[0] if sorted_rates else 0.0
    disparate_impact_ratio = (sorted_rates[0] / sorted_rates[-1]) if sorted_rates and sorted_rates[-1] > 0 else 1.0

    sorted_tprs = sorted(true_positive_rates.values())
    equal_opportunity_diff = sorted_tprs[-1] - sorted_tprs[0] if sorted_tprs else 0.0

    return {
        "demographic_parity_difference": demographic_parity_diff,
        "disparate_impact_ratio": disparate_impact_ratio,
        "equal_opportunity_difference": equal_opportunity_diff,
        "cohort_selection_rates": selection_rates
    }


# =====================================================================
# SECTION 2: ROBUST CROSS-VALIDATION MODEL ORCHESTRATION
# =====================================================================

def run_production_training_pipeline(sensitive_column_name: str = "Gender"):
    """
    EXPLANATION SECTION: Main execution block for model compilation.
    Runs 5-Fold Cross-Validation, builds full performance summary tables, 
    and prints out-of-fold detailed Classification Reports for deep audit analysis.
    """
    print(f"Starting Underwriting Model Training Pipeline: {datetime.now()}")
    
    df = load_clean_data()
    X, y = split_features_target(df)
    
    sensitive_feature = df[sensitive_column_name] if sensitive_column_name in df.columns else pd.Series(index=df.index, dtype=str)

    cat_cols, num_cols = get_feature_columns(X)

    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    candidate_models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    performance_records = []
    fairness_records = []
    detailed_reports = {}  # Injected to temporarily store out-of-fold reports in memory
    trained_pipelines = {}

    for model_name, classifier in candidate_models.items():
        print(f"Evaluating Model Candidate Profile: {model_name}...")
        
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        oof_predictions = np.zeros(len(X))
        oof_probabilities = np.zeros(len(X))
        fold_auc_scores = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_train, y_train)

            fold_preds = pipeline.predict(X_val)
            fold_probs = pipeline.predict_proba(X_val)[:, 1]

            oof_predictions[val_idx] = fold_preds
            oof_probabilities[val_idx] = fold_probs
            
            fold_auc_scores.append(roc_auc_score(y_val, fold_probs))

        final_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        final_pipeline.fit(X, y)
        trained_pipelines[model_name] = final_pipeline

        mean_cv_auc = np.mean(fold_auc_scores)
        overall_accuracy = accuracy_score(y, oof_predictions)
        tn, fp, fn, tp = confusion_matrix(y, oof_predictions).ravel()

        performance_records.append({
            "Model_Name": model_name,
            "CV_Mean_ROC_AUC": mean_cv_auc,
            "Accuracy": overall_accuracy,
            "True_Negatives": tn,
            "False_Positives": fp,
            "False_Negatives": fn,
            "True_Positives": tp
        })

        # INJECTED: Generate out-of-fold classification text reports for deep inspection
        detailed_reports[model_name] = classification_report(
            y, oof_predictions, target_names=["Rejected (0)", "Approved (1)"]
        )

        fairness = audit_fairness_metrics(y.values, oof_predictions, sensitive_feature)
        fairness_records.append({
            "Model_Name": model_name,
            "Demographic_Parity_Diff": fairness["demographic_parity_difference"],
            "Disparate_Impact_Ratio": fairness["disparate_impact_ratio"],
            "Equal_Opportunity_Diff": fairness["equal_opportunity_difference"]
        })

    summary_perf_df = pd.DataFrame(performance_records).sort_values(by="CV_Mean_ROC_AUC", ascending=False)
    summary_fair_df = pd.DataFrame(fairness_records)

    print("\n" + "="*60 + "\nPRODUCTION PERFORMANCE SUMMARY (OOF CV)\n" + "="*60)
    print(summary_perf_df.to_string(index=False))

    # INJECTED SECTION: Print complete, itemized Classification Reports for all classifiers
    print("\n" + "="*60 + "\nDETAILED AUDIT REPORTS PER MODEL CANDIDATE (OOF CV)\n" + "="*60)
    for model_name, report_text in detailed_reports.items():
        print(f"\n▶ Model: {model_name}")
        print("-" * 55)
        print(report_text)

    print("\n" + "="*60 + "\nFAIR LENDING REGULATORY COMPLIANCE REPORT\n" + "="*60)
    print(summary_fair_df.to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_perf_df.to_csv(os.path.join(OUTPUT_DIR, "model_performance_cv.csv"), index=False)
    summary_fair_df.to_csv(os.path.join(OUTPUT_DIR, "fairness_audit_report.csv"), index=False)

    best_model_name = summary_perf_df.iloc[0]["Model_Name"]
    best_pipeline = trained_pipelines[best_model_name]

    os.makedirs(MODEL_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_filename = f"pipeline_{best_model_name}_{timestamp}.joblib"
    
    joblib.dump(best_pipeline, os.path.join(MODEL_DIR, versioned_filename))
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    print(f"\nSuccessfully selected best model pipeline: {best_model_name}")
    print(f"Artifact deployed to production path: {BEST_MODEL_PATH}")
    print(f"Lineage backup saved to file: {versioned_filename}\n")


if __name__ == "__main__":
    run_production_training_pipeline(sensitive_column_name="Gender")