"""Uji determinisme, split berstrata, preprocessing, dan kebocoran data."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_loader import (
    _synthetic_dataset,
    inject_missing_values,
    load_dataset,
)
from src.preprocessing import build_preprocessor, remove_duplicates
from src.train_evaluate import evaluate_configuration

ROOT = Path(__file__).resolve().parents[1]


class MachineLearningTests(unittest.TestCase):
    """Memverifikasi kontrak utama pipeline data dan eksperimen."""

    @classmethod
    def setUpClass(cls) -> None:
        import yaml

        cls.config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8-sig"))

    def test_stratified_split_preserves_class_proportions(self) -> None:
        bundle = load_dataset({**self.config, "dataset": {
            **self.config["dataset"], "source": "synthetic",
            "fallback_samples": 500,
        }})
        x_train, x_test, y_train, y_test = train_test_split(
            bundle.features, bundle.target, test_size=0.2,
            stratify=bundle.target, random_state=11,
        )
        self.assertEqual(len(x_test), 100)
        self.assertLess(abs(y_train.mean() - bundle.target.mean()), 0.02)
        self.assertLess(abs(y_test.mean() - bundle.target.mean()), 0.02)

    def test_fit_preprocessor_does_not_use_test_values(self) -> None:
        train = pd.DataFrame({"nilai": [1.0, 3.0, 5.0]})
        test = pd.DataFrame({"nilai": [10_000.0]})
        preprocessor = build_preprocessor(train, "median", "standard", "onehot")
        pipeline = Pipeline([("preprocess", preprocessor)])
        pipeline.fit(train)
        scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
        self.assertAlmostEqual(float(scaler.mean_[0]), 3.0)
        self.assertNotEqual(float(scaler.mean_[0]), float(test["nilai"].mean()))

    def test_missing_injection_is_reproducible(self) -> None:
        bundle = _synthetic_dataset(self.config)
        first = inject_missing_values(bundle.features, [0.05, 0.20], seed=42)
        second = inject_missing_values(bundle.features, [0.05, 0.20], seed=42)
        for rate in first:
            pd.testing.assert_frame_equal(first[rate], second[rate])
        self.assertEqual(
            int(first[0.05].isna().any(axis=1).sum()),
            round(len(first[0.05]) * 0.05),
        )
        self.assertTrue((first[0.05].isna().sum(axis=1) <= 1).all())

    def test_duplicate_rows_are_removed_without_target_misalignment(self) -> None:
        features = pd.DataFrame({"x": [1, 2, 1], "category": ["a", "b", "a"]})
        target = pd.Series([0, 1, 0])
        clean_x, clean_y, counts = remove_duplicates(features, target)
        self.assertEqual(len(clean_x), 2)
        self.assertEqual(clean_y.tolist(), [0, 1])
        self.assertEqual(counts["duplicates_before_removal"], 1)
        self.assertEqual(counts["duplicates_after_removal"], 0)

    def test_repeated_model_evaluation_is_seeded(self) -> None:
        bundle = _synthetic_dataset({**self.config, "dataset": {
            **self.config["dataset"], "fallback_samples": 180,
            "fallback_features": 6, "fallback_informative": 4,
            "fallback_redundant": 1,
        }})
        x = bundle.features
        y = bundle.target
        scenario = {
            "name": "uji_deterministik", "missing": "median",
            "scaler": "standard", "categorical_encoder": "onehot",
        }
        arguments = dict(
            features=x, target=y, scenario=scenario,
            algorithm="logistic_regression",
            algorithm_settings={"max_iter": 300, "C": 1.0, "class_weight": None},
            seed=37, test_size=0.2, cv_folds=3,
            positive_label=y.value_counts().idxmin(),
            overfit_gap_threshold=0.1, near_perfect_accuracy=0.98,
        )
        first = evaluate_configuration(**arguments)
        second = evaluate_configuration(**arguments)
        for metric in ("test_accuracy", "test_recall", "test_f1", "test_roc_auc"):
            self.assertEqual(first[metric], second[metric])

    def test_drop_missing_scenario_runs(self) -> None:
        bundle = _synthetic_dataset({**self.config, "dataset": {
            **self.config["dataset"], "fallback_samples": 180,
            "fallback_features": 6, "fallback_informative": 4,
            "fallback_redundant": 1,
        }})
        features = inject_missing_values(bundle.features, [0.05], 4)[0.05]
        scenario = {
            "name": "uji_hapus", "missing": "drop", "scaler": "none",
            "categorical_encoder": "ordinal",
        }
        result = evaluate_configuration(
            features=features, target=bundle.target, scenario=scenario,
            algorithm="decision_tree",
            algorithm_settings={"max_depth": 3, "min_samples_leaf": 2, "class_weight": None},
            seed=11, test_size=0.2, cv_folds=3,
            positive_label=bundle.target.value_counts().idxmin(),
            overfit_gap_threshold=0.1, near_perfect_accuracy=0.98,
        )
        self.assertGreaterEqual(result["test_f1"], 0.0)
        self.assertLessEqual(result["test_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
