"""Model definitions, hyperparameter tuning, cross-validation, and persistence.

Implements strict featurization ordering (preprocessors fit strictly on train folds)
and provides systematic multi-model comparison.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiate candidate classifiers for comparison.

    Includes:
    - Logistic Regression (linear probabilistic baseline)
    - K-Nearest Neighbors (instance-based non-parametric)
    - Decision Tree (interpretable rule-based)
    - Random Forest (ensemble bagging)
    - Support Vector Machine (maximum margin with RBF kernel)
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=500,
            random_state=random_state,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=random_state,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
        ),
    }


def build_pipeline(classifier: Any) -> Pipeline:
    """Wrap a classifier into a Pipeline with StandardScaler.

    Guarantees strict featurization ordering (scaler is fitted exclusively on
    the training folds during CV and train set during final fitting, avoiding leakage).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ])


def evaluate_cross_validation(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """Run Stratified K-Fold Cross-Validation on all models.

    Computes mean and std for Accuracy, Precision, Recall, and F1-Score.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    results = []
    for name, clf in models.items():
        pipe = build_pipeline(clf)
        cv_res = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=skf,
            scoring=scoring,
            return_train_score=True,
        )

        results.append({
            "Model": name,
            "CV Accuracy (Mean)": cv_res["test_accuracy"].mean(),
            "CV Accuracy (Std)": cv_res["test_accuracy"].std(),
            "CV F1-Macro (Mean)": cv_res["test_f1_macro"].mean(),
            "CV Precision (Mean)": cv_res["test_precision_macro"].mean(),
            "CV Recall (Mean)": cv_res["test_recall_macro"].mean(),
            "Train Accuracy": cv_res["train_accuracy"].mean(),
            "Fit Time (ms)": cv_res["fit_time"].mean() * 1000,
        })

    summary_df = pd.DataFrame(results).sort_values(by="CV Accuracy (Mean)", ascending=False)
    return summary_df


def tune_best_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_k: int = 20,
) -> Tuple[int, Dict[int, float]]:
    """Determine the optimal k parameter for KNN via cross-validation."""
    k_range = list(range(1, max_k + 1))
    param_grid = {"classifier__n_neighbors": k_range}
    pipe = build_pipeline(KNeighborsClassifier())

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    grid.fit(X_train, y_train)

    k_scores = dict(zip(k_range, grid.cv_results_["mean_test_score"]))
    best_k = grid.best_params_["classifier__n_neighbors"]
    return best_k, k_scores


def fit_and_save_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: str = "models",
) -> Dict[str, Pipeline]:
    """Fit all pipelines on training set and serialize models to disk."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fitted_pipelines = {}
    for name, clf in models.items():
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Standardize safe filename
        slug = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(pipe, out_dir / f"{slug}_pipeline.joblib")

    return fitted_pipelines


def save_best_model(
    best_model_name: str,
    fitted_pipelines: Dict[str, Pipeline],
    feature_names: list,
    target_names: list,
    output_dir: str = "models",
) -> Path:
    """Export the designated top-performing pipeline with metadata."""
    out_dir = Path(output_dir)
    best_pipe = fitted_pipelines[best_model_name]

    # Save top model
    best_model_path = out_dir / "best_model.joblib"
    joblib.dump(best_pipe, best_model_path)

    # Save metadata dictionary for standalone inference
    metadata = {
        "model_name": best_model_name,
        "feature_names": feature_names,
        "target_names": target_names,
    }
    joblib.dump(metadata, out_dir / "model_metadata.joblib")

    return best_model_path
