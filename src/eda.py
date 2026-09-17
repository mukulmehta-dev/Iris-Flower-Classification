"""Exploratory Data Analysis (EDA) module for the Iris dataset.

Generates high-resolution visualization artifacts:
- Feature distribution histograms and KDEs
- Pairwise feature relationship plots (pairplots)
- Correlation heatmaps
- 2D PCA projection showcasing linear separability and class boundaries
- Box plots highlighting inter-quartile ranges and species variation
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Modern styling constants
SPECIES_PALETTE = {
    "setosa": "#6366F1",     # Indigo
    "versicolor": "#EC4899", # Pink/Rose
    "virginica": "#10B981",  # Emerald
}

FEATURE_LABELS = {
    "sepal_length": "Sepal Length (cm)",
    "sepal_width": "Sepal Width (cm)",
    "petal_length": "Petal Length (cm)",
    "petal_width": "Petal Width (cm)",
}


def setup_style() -> None:
    """Set clean modern aesthetics for matplotlib/seaborn."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 200,
    })


def plot_feature_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot histograms and KDEs for all four features grouped by species."""
    setup_style()
    features = list(FEATURE_LABELS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, feat in enumerate(features):
        ax = axes[idx]
        for species, color in SPECIES_PALETTE.items():
            subset = df[df["species"] == species][feat]
            sns.kdeplot(
                data=subset,
                ax=ax,
                label=species.capitalize(),
                color=color,
                fill=True,
                alpha=0.3,
                linewidth=2,
            )
        ax.set_title(f"Distribution of {FEATURE_LABELS[feat]}")
        ax.set_xlabel(FEATURE_LABELS[feat])
        ax.set_ylabel("Density")
        ax.legend(title="Species", frameon=True)

    plt.suptitle("Feature Density Distributions across Iris Species", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "feature_distributions.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pairplot(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create comprehensive pairplot of all feature interactions."""
    setup_style()
    plot_df = df.copy()
    plot_df["species"] = plot_df["species"].str.capitalize()
    renamed_cols = {col: FEATURE_LABELS[col] for col in FEATURE_LABELS}
    plot_df = plot_df.rename(columns=renamed_cols)

    palette = {k.capitalize(): v for k, v in SPECIES_PALETTE.items()}

    pair_grid = sns.pairplot(
        plot_df,
        hue="species",
        vars=list(FEATURE_LABELS.values()),
        palette=palette,
        diag_kind="kde",
        plot_kws={"alpha": 0.8, "s": 45, "edgecolor": "none"},
        height=2.4,
    )
    pair_grid.fig.suptitle("Iris Pairwise Feature Scatter Matrix & Distributions", y=1.02)
    out_path = output_dir / "pairplot.png"
    pair_grid.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot Pearson correlation matrix between morphological measurements."""
    setup_style()
    features = list(FEATURE_LABELS.keys())
    corr = df[features].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    clean_labels = [FEATURE_LABELS[f] for f in features]
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=-1,
        vmax=1,
        linewidths=1,
        square=True,
        xticklabels=clean_labels,
        yticklabels=clean_labels,
        ax=ax,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
    )
    ax.set_title("Iris Feature Correlation Matrix")
    plt.tight_layout()
    out_path = output_dir / "correlation_heatmap.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pca_projection(df: pd.DataFrame, output_dir: Path) -> Path:
    """Project dataset into 2D PCA space to illustrate class clustering."""
    setup_style()
    features = list(FEATURE_LABELS.keys())
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    for species, color in SPECIES_PALETTE.items():
        mask = df["species"] == species
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=color,
            label=f"Iris {species.capitalize()}",
            s=70,
            alpha=0.85,
            edgecolors="w",
            linewidth=0.8,
        )

    ax.set_title(
        f"2D Principal Component Analysis (PCA)\nExplains {sum(evr):.1%} Total Variance (PC1: {evr[0]:.1%}, PC2: {evr[1]:.1%})"
    )
    ax.set_xlabel(f"Principal Component 1 ({evr[0]:.1%} variance)")
    ax.set_ylabel(f"Principal Component 2 ({evr[1]:.1%} variance)")
    ax.legend(title="Species", frameon=True, loc="upper right")
    plt.tight_layout()
    out_path = output_dir / "pca_projection.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_boxplots(df: pd.DataFrame, output_dir: Path) -> Path:
    """Plot box plots showing median, IQR, and variance per species."""
    setup_style()
    features = list(FEATURE_LABELS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, feat in enumerate(features):
        ax = axes[idx]
        sns.boxplot(
            data=df,
            x="species",
            y=feat,
            palette=SPECIES_PALETTE,
            hue="species",
            legend=False,
            ax=ax,
            boxprops=dict(alpha=0.8),
            width=0.45,
        )
        sns.stripplot(
            data=df,
            x="species",
            y=feat,
            color="black",
            alpha=0.35,
            size=4,
            jitter=0.2,
            ax=ax,
        )
        ax.set_title(f"{FEATURE_LABELS[feat]} by Species")
        ax.set_xlabel("")
        ax.set_ylabel(FEATURE_LABELS[feat])
        ax.set_xticklabels([s.capitalize() for s in SPECIES_PALETTE.keys()])

    plt.suptitle("Feature Variations and Outlier Inspection by Species", y=1.01)
    plt.tight_layout()
    out_path = output_dir / "boxplots.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_full_eda(df: pd.DataFrame, output_dir: str = "reports/figures") -> Dict[str, Path]:
    """Run all EDA routines and save plots to the output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = {
        "distributions": plot_feature_distributions(df, out_dir),
        "pairplot": plot_pairplot(df, out_dir),
        "correlation": plot_correlation_heatmap(df, out_dir),
        "pca": plot_pca_projection(df, out_dir),
        "boxplots": plot_boxplots(df, out_dir),
    }
    return generated
