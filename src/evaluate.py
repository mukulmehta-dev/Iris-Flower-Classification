"""Evaluation, diagnostic plotting, and metrics calculation module.

Generates:
- Multi-model confusion matrix heatmaps
- Multiclass One-vs-Rest ROC curves with AUC
- 2D Decision boundary contour maps comparing linear, tree, and kernel boundaries
- Feature importance rankings
- KNN hyperparameter elbow plot
"""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.eda import setup_style, SPECIES_PALETTE


def compute_test_metrics(
    fitted_pipelines: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compute comprehensive test-set classification metrics for all models."""
    rows = []
    for name, pipe in fitted_pipelines.items():
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC AUC if probability estimation supported
        roc_auc = np.nan
        if hasattr(pipe, "predict_proba"):
            try:
                y_prob = pipe.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            except Exception:
                pass

        rows.append({
            "Model": name,
            "Test Accuracy": acc,
            "Precision (Macro)": prec,
            "Recall (Macro)": rec,
            "F1-Score (Macro)": f1,
            "F1-Score (Weighted)": f1_weighted,
            "ROC-AUC (Macro)": roc_auc,
        })

    df_metrics = pd.DataFrame(rows).sort_values(by="Test Accuracy", ascending=False)
    return df_metrics


def plot_confusion_matrices(
    fitted_pipelines: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: List[str],
    output_dir: Path,
) -> Path:
    """Plot multi-panel confusion matrices for all evaluated models."""
    setup_style()
    n_models = len(fitted_pipelines)
    cols = 3
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for idx, (name, pipe) in enumerate(fitted_pipelines.items()):
        ax = axes[idx]
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Format labels with counts and normalized percentage
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        labels = [
            [f"{val}\n({cm_norm[r, c]:.0%})" for c, val in enumerate(row)]
            for r, row in enumerate(cm)
        ]

        sns.heatmap(
            cm,
            annot=labels,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=[t.capitalize() for t in target_names],
            yticklabels=[t.capitalize() for t in target_names],
            ax=ax,
            square=True,
        )
        acc = accuracy_score(y_test, y_pred)
        ax.set_title(f"{name}\nAccuracy: {acc:.2%}")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    # Hide extra unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Test Set Confusion Matrices Across Evaluated Models", y=1.02)
    plt.tight_layout()
    out_path = output_dir / "confusion_matrices.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_multiclass_roc(
    fitted_pipelines: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: List[str],
    output_dir: Path,
) -> Path:
    """Plot One-vs-Rest ROC curves for models with probabilistic output."""
    setup_style()
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for class_idx in range(n_classes):
        ax = axes[class_idx]
        species_name = target_names[class_idx].capitalize()

        for name, pipe in fitted_pipelines.items():
            if not hasattr(pipe, "predict_proba"):
                continue
            y_prob = pipe.predict_proba(X_test)[:, class_idx]
            fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_prob)
            auc = roc_auc_score(y_bin[:, class_idx], y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
        ax.set_title(f"ROC: Iris {species_name} vs Rest")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)

    plt.suptitle("One-vs-Rest ROC Curves by Species", y=1.03)
    plt.tight_layout()
    out_path = output_dir / "roc_curves.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_decision_boundaries(
    fitted_pipelines: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_names: List[str],
    output_dir: Path,
) -> Path:
    """Visualize 2D decision boundary contours for Petal Length vs Petal Width.

    Petal measurements provide the highest discriminatory signal.
    """
    setup_style()
    # Use petal dimensions for 2D visualization
    feature_x = "petal_length"
    feature_y = "petal_width"
    
    # We retrain simple 2-feature versions of each classifier strictly for the 2D contour plot
    x_min, x_max = X_train[feature_x].min() - 0.5, X_train[feature_x].max() + 0.5
    y_min, y_max = X_train[feature_y].min() - 0.5, X_train[feature_y].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_points = pd.DataFrame({feature_x: xx.ravel(), feature_y: yy.ravel()})

    models_to_plot = ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Support Vector Machine"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    colors = ["#c7d2fe", "#fbcfe8", "#a7f3d0"]  # Soft pastel fills matching species
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(colors)

    for idx, model_name in enumerate(models_to_plot):
        if model_name not in fitted_pipelines:
            continue
        ax = axes[idx]
        
        # Fit 2D pipeline on petal features
        from sklearn.base import clone
        base_clf = clone(fitted_pipelines[model_name].named_steps["classifier"])
        from src.models import build_pipeline
        pipe_2d = build_pipeline(base_clf)
        pipe_2d.fit(X_train[[feature_x, feature_y]], y_train)

        Z = pipe_2d.predict(grid_points).reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
        ax.contour(xx, yy, Z, colors="grey", linewidths=0.7, linestyles="dashed")

        # Overlay scatter points
        for c_idx, species in enumerate(target_names):
            mask = y_train == c_idx
            ax.scatter(
                X_train.loc[mask, feature_x],
                X_train.loc[mask, feature_y],
                c=list(SPECIES_PALETTE.values())[c_idx],
                label=species.capitalize(),
                edgecolors="white",
                linewidth=0.8,
                s=55,
            )

        ax.set_title(f"Decision Boundary: {model_name}")
        ax.set_xlabel("Petal Length (cm)")
        ax.set_ylabel("Petal Width (cm)")
        ax.legend(title="Species", loc="upper left")

    plt.suptitle("Comparison of 2D Classifier Decision Boundaries (Petal Features)", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "decision_boundaries.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_knn_elbow(k_scores: Dict[int, float], best_k: int, output_dir: Path) -> Path:
    """Plot cross-validation accuracy vs K values to visually justify optimal K."""
    setup_style()
    ks = list(k_scores.keys())
    scores = list(k_scores.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, scores, marker="o", color="#6366F1", linewidth=2.2, markersize=6)
    ax.axvline(best_k, color="#EF4444", linestyle="--", linewidth=1.5, label=f"Optimal k={best_k} ({k_scores[best_k]:.3f})")

    ax.set_title("KNN Hyperparameter Optimization (Elbow Analysis)")
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("Cross-Validation Accuracy")
    ax.set_xticks(ks)
    ax.legend(frameon=True)
    plt.tight_layout()

    out_path = output_dir / "knn_elbow_curve.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_importances(
    fitted_pipelines: Dict[str, Any],
    feature_names: List[str],
    output_dir: Path,
) -> Path:
    """Plot feature importance ranking from tree-based ensembles."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    tree_models = [("Decision Tree", axes[0]), ("Random Forest", axes[1])]
    from src.eda import FEATURE_LABELS

    for name, ax in tree_models:
        if name in fitted_pipelines:
            clf = fitted_pipelines[name].named_steps["classifier"]
            importances = clf.feature_importances_
            clean_names = [FEATURE_LABELS[f] for f in feature_names]
            
            sorted_idx = np.argsort(importances)
            ax.barh(
                np.array(clean_names)[sorted_idx],
                importances[sorted_idx],
                color="#EC4899" if name == "Decision Tree" else "#10B981",
                alpha=0.85,
            )
            ax.set_title(f"{name} Feature Importance")
            ax.set_xlabel("Relative Importance Gini Index")

    plt.suptitle("Predictive Power by Feature (Gini Importance)", y=1.02)
    plt.tight_layout()
    out_path = output_dir / "feature_importance.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
