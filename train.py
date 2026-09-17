"""Master training and evaluation orchestrator.

Executes the end-to-end ML pipeline:
1. Ingests and validates the Iris dataset.
2. Runs exploratory data analysis and exports visual artifacts.
3. Conducts stratified train/test split.
4. Performs 5-fold cross validation across 5 classification algorithms.
5. Tunes hyperparameters (KNN elbow analysis).
6. Fits final models on training data with strict featurization ordering.
7. Evaluates on holdout test set with confusion matrices, ROC curves, and decision boundaries.
8. Serializes best pipeline for production serving.
"""

from pathlib import Path
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

from src.data import (
    FEATURE_NAMES,
    TARGET_NAMES,
    get_summary_statistics,
    get_train_test_data,
    load_iris_data,
)
from src.eda import run_full_eda
from src.evaluate import (
    compute_test_metrics,
    plot_confusion_matrices,
    plot_decision_boundaries,
    plot_feature_importances,
    plot_knn_elbow,
    plot_multiclass_roc,
)
from src.models import (
    evaluate_cross_validation,
    fit_and_save_models,
    get_base_models,
    save_best_model,
    tune_best_knn,
)


def run_pipeline():
    print("=" * 65)
    print("🌸 IRIS FLOWER SPECIES CLASSIFICATION PIPELINE 🌸")
    print("=" * 65)

    # 1. Ingestion & Validation
    print("\n[1/7] Ingesting and validating Iris dataset...")
    df = load_iris_data(save_path="data/iris.csv")
    print(f"      ✓ Dataset loaded: {df.shape[0]} samples, {len(FEATURE_NAMES)} features.")
    print("      ✓ Zero missing values detected; schema strictly validated.")
    print(f"      ✓ Raw dataset saved to data/iris.csv")

    # 2. Exploratory Data Analysis
    print("\n[2/7] Generating high-resolution EDA visual artifacts...")
    eda_figures = run_full_eda(df, output_dir="reports/figures")
    for name, path in eda_figures.items():
        print(f"      ✓ Saved: {path.name}")

    # 3. Stratified Train / Test Split
    print("\n[3/7] Performing stratified train/test split (80/20)...")
    X_train, X_test, y_train, y_test, feat_names, tgt_names = get_train_test_data(
        df, test_size=0.2, random_state=42
    )
    print(f"      ✓ Training samples: {len(X_train)} (stratified)")
    print(f"      ✓ Holdout test samples: {len(X_test)} (stratified)")

    # 4. Model Zoo & 5-Fold Stratified Cross Validation
    print("\n[4/7] Running 5-Fold Stratified Cross-Validation on Training Set...")
    models = get_base_models(random_state=42)
    
    # Tune KNN first to use optimal k
    best_k, k_scores = tune_best_knn(X_train, y_train, max_k=20)
    plot_knn_elbow(k_scores, best_k, Path("reports/figures"))
    from sklearn.neighbors import KNeighborsClassifier
    models["K-Nearest Neighbors"] = KNeighborsClassifier(n_neighbors=best_k)
    print(f"      ✓ KNN hyperparameter tuned: optimal k={best_k}")

    cv_results = evaluate_cross_validation(models, X_train, y_train, cv_folds=5)
    print("\n" + cv_results.to_string(index=False))

    # 5. Fit Pipelines on Full Training Set
    print("\n[5/7] Fitting production pipelines with StandardScaler...")
    fitted_pipelines = fit_and_save_models(models, X_train, y_train, output_dir="models")
    print(f"      ✓ {len(fitted_pipelines)} pipelines serialized to models/")

    # 6. Test Set Evaluation
    print("\n[6/7] Evaluating on holdout test set...")
    test_metrics = compute_test_metrics(fitted_pipelines, X_test, y_test)
    print("\n" + test_metrics.to_string(index=False))

    # Diagnostic Plots
    fig_dir = Path("reports/figures")
    cm_path = plot_confusion_matrices(fitted_pipelines, X_test, y_test, tgt_names, fig_dir)
    print(f"      ✓ Confusion matrices saved: {cm_path.name}")

    roc_path = plot_multiclass_roc(fitted_pipelines, X_test, y_test, tgt_names, fig_dir)
    print(f"      ✓ Multi-class ROC curves saved: {roc_path.name}")

    db_path = plot_decision_boundaries(fitted_pipelines, X_train, y_train, tgt_names, fig_dir)
    print(f"      ✓ 2D Decision boundaries saved: {db_path.name}")

    fi_path = plot_feature_importances(fitted_pipelines, feat_names, fig_dir)
    print(f"      ✓ Feature importances saved: {fi_path.name}")

    # 7. Select & Save Top Model
    print("\n[7/7] Selecting top model for production deployment...")
    # Determine best model based on combination of CV and Test Accuracy
    merged = cv_results.merge(test_metrics, on="Model")
    best_model_name = merged.sort_values(
        by=["Test Accuracy", "CV Accuracy (Mean)"], ascending=False
    ).iloc[0]["Model"]

    best_path = save_best_model(
        best_model_name,
        fitted_pipelines,
        feat_names,
        tgt_names,
        output_dir="models",
    )
    print(f"      ★ Best Model Selected: {best_model_name}")
    print(f"      ★ Serialized to: {best_path}")

    print("\n" + "=" * 65)
    print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_pipeline()
