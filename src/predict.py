"""Production inference pipeline and CLI for Iris Flower Classification.

Loads serialized model pipelines and provides single-sample, batch, and CLI inference.
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
import pandas as pd

from src.data import FEATURE_NAMES, TARGET_NAMES


class IrisPredictor:
    """Inference engine for Iris flower species classification."""

    def __init__(
        self,
        model_path: str = "models/best_model.joblib",
        metadata_path: str = "models/model_metadata.joblib",
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. Please run 'train.py' first."
            )

        self.pipeline = joblib.load(self.model_path)
        
        if self.metadata_path.exists():
            self.metadata = joblib.load(self.metadata_path)
            self.target_names = self.metadata.get("target_names", TARGET_NAMES)
            self.model_name = self.metadata.get("model_name", "Classifier")
        else:
            self.target_names = TARGET_NAMES
            self.model_name = "Iris Classifier"

    def predict_one(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> Dict[str, Any]:
        """Classify a single iris specimen given its dimensions in centimeters.

        Parameters
        ----------
        sepal_length : float
        sepal_width : float
        petal_length : float
        petal_width : float

        Returns
        -------
        Dict containing predicted class, confidence, and probability distribution.
        """
        input_df = pd.DataFrame(
            [[sepal_length, sepal_width, petal_length, petal_width]],
            columns=FEATURE_NAMES,
        )

        pred_idx = int(self.pipeline.predict(input_df)[0])
        pred_species = self.target_names[pred_idx]

        probabilities = {}
        confidence = 1.0
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(input_df)[0]
            confidence = float(probs[pred_idx])
            probabilities = {
                self.target_names[i]: float(probs[i])
                for i in range(len(self.target_names))
            }

        return {
            "predicted_species": pred_species,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_used": self.model_name,
            "inputs": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
            },
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run batch predictions on an arbitrary DataFrame containing the 4 features."""
        df_clean = df[FEATURE_NAMES].copy()
        preds = self.pipeline.predict(df_clean)
        result_df = df.copy()
        result_df["predicted_target"] = preds
        result_df["predicted_species"] = [self.target_names[i] for i in preds]

        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(df_clean)
            for idx, name in enumerate(self.target_names):
                result_df[f"prob_{name}"] = probs[:, idx]
            result_df["confidence"] = np.max(probs, axis=1)

        return result_df


def main():
    """CLI entrypoint for running iris predictions."""
    parser = argparse.ArgumentParser(description="Classify an Iris flower from measurements.")
    parser.add_argument("--sepal-length", type=float, required=True, help="Sepal length in cm")
    parser.add_argument("--sepal-width", type=float, required=True, help="Sepal width in cm")
    parser.add_argument("--petal-length", type=float, required=True, help="Petal length in cm")
    parser.add_argument("--petal-width", type=float, required=True, help="Petal width in cm")
    parser.add_argument("--model-path", type=str, default="models/best_model.joblib", help="Path to .joblib model")

    args = parser.parse_args()

    predictor = IrisPredictor(model_path=args.model_path)
    result = predictor.predict_one(
        args.sepal_length,
        args.sepal_width,
        args.petal_length,
        args.petal_width,
    )

    print("\n" + "=" * 45)
    print(f"🌸 Prediction Result: Iris {result['predicted_species'].capitalize()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Model Architecture: {result['model_used']}")
    print("-" * 45)
    print("Class Probabilities:")
    for species, prob in result["probabilities"].items():
        bar = "█" * int(prob * 25)
        print(f"  • {species.capitalize():<12}: {prob:6.2%} {bar}")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
