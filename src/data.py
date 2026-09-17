"""Data loading, validation, and splitting module for Iris dataset."""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Standardized feature column names
FEATURE_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_NAMES = ["setosa", "versicolor", "virginica"]


def load_iris_data(save_path: str = "data/iris.csv") -> pd.DataFrame:
    """Load the Iris dataset, validate schema and integrity, and save locally.

    Parameters
    ----------
    save_path : str
        Relative or absolute path to save the raw CSV copy.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame containing features, numerical target, and species name.
    """
    raw_data = load_iris(as_frame=True)
    df = raw_data.frame.copy()

    # Rename columns to clean, pythonic names
    rename_mapping = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "target",
    }
    df = df.rename(columns=rename_mapping)

    # Map numeric target to species name
    target_names_dict = {i: name for i, name in enumerate(raw_data.target_names)}
    df["species"] = df["target"].map(target_names_dict)

    # Data Integrity & Schema Validation
    validate_dataset(df)

    # Persist locally if directory is specified
    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Perform strict data integrity checks on the Iris dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataframe to validate.

    Raises
    ------
    ValueError
        If schema is invalid, missing values exist, or values violate domain rules.
    """
    expected_cols = set(FEATURE_NAMES + ["target", "species"])
    if not expected_cols.issubset(set(df.columns)):
        missing = expected_cols - set(df.columns)
        raise ValueError(f"Dataset missing expected columns: {missing}")

    # Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        raise ValueError(f"Dataset contains unexpected NULL/NaN values:\n{null_counts[null_counts > 0]}")

    # Check for physical validity (all measurements must be strictly positive)
    for col in FEATURE_NAMES:
        if (df[col] <= 0).any():
            raise ValueError(f"Column '{col}' contains non-positive values!")

    # Validate target classes
    unique_targets = sorted(df["target"].unique())
    if unique_targets != [0, 1, 2]:
        raise ValueError(f"Expected targets [0, 1, 2], found {unique_targets}")


def get_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """Perform a stratified train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Validated Iris dataset.
    test_size : float
        Proportion of dataset to include in test split (default: 0.2).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Tuple
        X_train, X_test, y_train, y_test, feature_names, target_names
    """
    X = df[FEATURE_NAMES]
    y = df["target"]

    # Stratified split ensures balanced class distribution in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, FEATURE_NAMES, TARGET_NAMES


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate summary statistics per species for the features."""
    return df.groupby("species")[FEATURE_NAMES].agg(["count", "mean", "std", "min", "median", "max"])
