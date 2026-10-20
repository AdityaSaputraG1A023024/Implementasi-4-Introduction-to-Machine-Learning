"""Menjalankan algoritma klasifikasi dan mengukur hasil lintas random seed."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor, split_drop_missing

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def build_model(name: str, settings: dict[str, Any], seed: int) -> Any:
    """Membuat model sesuai nama dan hiperparameter yang dicatat di config."""
    if name == "majority_baseline":
        return DummyClassifier(strategy="most_frequent")
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=int(settings["max_iter"]),
            C=float(settings["C"]),
            class_weight=settings.get("class_weight"),
            random_state=seed,
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=settings.get("max_depth"),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            class_weight=settings.get("class_weight"),
            random_state=seed,
        )
    if name == "svm":
        return SVC(
            C=float(settings["C"]),
            kernel=str(settings["kernel"]),
            class_weight=settings.get("class_weight"),
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(settings["n_estimators"]),
            max_depth=settings.get("max_depth"),
            min_samples_leaf=int(settings["min_samples_leaf"]),
            class_weight=settings.get("class_weight"),
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Algoritma tidak dikenal: {name}.")


def _metrics(y_true: pd.Series, y_pred: np.ndarray, probabilities: np.ndarray, positive_label: Any) -> dict[str, float]:
    """Menghitung metrik biner dengan label positif minoritas yang konsisten."""
    truth = np.asarray(y_true)
    positive_truth = (truth == positive_label).astype(int)
    positive_pred = (np.asarray(y_pred) == positive_label).astype(int)
    try:
        auc = float(roc_auc_score(positive_truth, probabilities))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(truth, y_pred)),
        "precision": float(precision_score(positive_truth, positive_pred, zero_division=0)),
        "recall": float(recall_score(positive_truth, positive_pred, zero_division=0)),
        "f1": float(f1_score(positive_truth, positive_pred, zero_division=0)),
        "roc_auc": auc,
    }


def _positive_probability(estimator: Any, features: Any, positive_label: Any) -> np.ndarray:
    """Mengambil probabilitas atau skor keputusan label minoritas untuk ROC-AUC."""
    labels = list(estimator.classes_)
    if positive_label not in labels:
        return np.zeros(len(features), dtype=float)
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, labels.index(positive_label)]
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(features), dtype=float)
        return scores if labels.index(positive_label) == 1 else -scores
    return np.zeros(len(features), dtype=float)


def evaluate_configuration(
    *,
    features: pd.DataFrame,
    target: pd.Series,
    scenario: dict[str, str],
    algorithm: str,
    algorithm_settings: dict[str, Any],
    seed: int,
    test_size: float,
    cv_folds: int,
    positive_label: Any,
    overfit_gap_threshold: float,
    near_perfect_accuracy: float,
) -> dict[str, Any]:
    """Melakukan split, CV data latih, fit pipeline, dan evaluasi holdout."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        stratify=target,
        random_state=seed,
    )
    if scenario["missing"] == "drop":
        x_train, y_train, x_test, y_test = split_drop_missing(
            x_train, y_train, x_test, y_test
        )
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Partisi latih/uji harus memiliki kedua kelas untuk evaluasi.")

    preprocessor = build_preprocessor(
        x_train,
        scenario["missing"],
        scenario["scaler"],
        scenario["categorical_encoder"],
    )
    model = build_model(algorithm, algorithm_settings, seed)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_scoring: dict[str, str] = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        cv_result = cross_validate(
            clone(pipeline),
            x_train,
            y_train,
            cv=cv,
            scoring=cv_scoring,
            return_train_score=False,
            n_jobs=1,
            error_score="raise",
        )
        pipeline.fit(x_train, y_train)

    y_train_pred = pipeline.predict(x_train)
    y_test_pred = pipeline.predict(x_test)
    train_probability = _positive_probability(pipeline, x_train, positive_label)
    test_probability = _positive_probability(pipeline, x_test, positive_label)
    train_metrics = _metrics(y_train, y_train_pred, train_probability, positive_label)
    test_metrics = _metrics(y_test, y_test_pred, test_probability, positive_label)
    train_confusion = confusion_matrix(y_train == positive_label, y_train_pred == positive_label, labels=[False, True])
    test_confusion = confusion_matrix(y_test == positive_label, y_test_pred == positive_label, labels=[False, True])
    gap = train_metrics["f1"] - test_metrics["f1"]
    near_perfect = (
        train_metrics["accuracy"] >= near_perfect_accuracy
        or test_metrics["accuracy"] >= near_perfect_accuracy
    )
    overfit = gap > overfit_gap_threshold or near_perfect
    overfit_reasons = []
    if gap > overfit_gap_threshold:
        overfit_reasons.append("selisih_f1_train_uji")
    if near_perfect:
        overfit_reasons.append("akurasi_mendekati_sempurna_perlu_ditinjau")

    row: dict[str, Any] = {
        "scenario": scenario["name"],
        "missing_strategy": scenario["missing"],
        "scaler": scenario["scaler"],
        "categorical_encoder": scenario["categorical_encoder"],
        "algorithm": algorithm,
        "seed": seed,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "minority_label": str(positive_label),
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "test_precision": test_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "test_recall": test_metrics["recall"],
        "train_f1": train_metrics["f1"],
        "test_f1": test_metrics["f1"],
        "train_roc_auc": train_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "train_test_f1_gap": gap,
        "overfit_flag": bool(overfit),
        "near_perfect_accuracy_flag": bool(near_perfect),
        "overfit_reason": ";".join(overfit_reasons),
        "train_confusion_tn": int(train_confusion[0, 0]),
        "train_confusion_fp": int(train_confusion[0, 1]),
        "train_confusion_fn": int(train_confusion[1, 0]),
        "train_confusion_tp": int(train_confusion[1, 1]),
        "test_confusion_tn": int(test_confusion[0, 0]),
        "test_confusion_fp": int(test_confusion[0, 1]),
        "test_confusion_fn": int(test_confusion[1, 0]),
        "test_confusion_tp": int(test_confusion[1, 1]),
        "cv_accuracy_mean": float(np.mean(cv_result["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_result["test_accuracy"], ddof=1)),
        "cv_precision_macro_mean": float(np.mean(cv_result["test_precision"])),
        "cv_precision_macro_std": float(np.std(cv_result["test_precision"], ddof=1)),
        "cv_recall_macro_mean": float(np.mean(cv_result["test_recall"])),
        "cv_recall_macro_std": float(np.std(cv_result["test_recall"], ddof=1)),
        "cv_f1_macro_mean": float(np.mean(cv_result["test_f1"])),
        "cv_f1_macro_std": float(np.std(cv_result["test_f1"], ddof=1)),
    }
    for metric in METRICS:
        row[f"{metric}_test"] = test_metrics[metric]
    return row


def evaluate_experiment(
    *,
    features: pd.DataFrame,
    target: pd.Series,
    scenarios: list[dict[str, str]],
    algorithms: dict[str, dict[str, Any]],
    seeds: list[int],
    test_size: float,
    cv_folds: int,
    overfit_gap_threshold: float,
    near_perfect_accuracy: float,
) -> pd.DataFrame:
    """Mengulang semua pasangan skenario/model untuk seluruh seed terpilih."""
    class_counts = target.value_counts()
    positive_label = class_counts.idxmin()
    model_specs = {"majority_baseline": {}}
    model_specs.update(algorithms)
    rows: list[dict[str, Any]] = []
    total = len(scenarios) * len(model_specs) * len(seeds)
    completed = 0
    for scenario in scenarios:
        for algorithm, settings in model_specs.items():
            for seed in seeds:
                completed += 1
                if scenario["missing"] == "drop":
                    # Jika kelas minoritas terlalu kecil setelah drop, tetap gagalkan alih-alih membentuk split tidak valid.
                    min_class = int(class_counts.min())
                    if min_class < cv_folds + 1:
                        raise ValueError("Terlalu sedikit contoh kelas untuk drop dan cross-validation.")
                rows.append(evaluate_configuration(
                    features=features,
                    target=target,
                    scenario=scenario,
                    algorithm=algorithm,
                    algorithm_settings=settings,
                    seed=seed,
                    test_size=test_size,
                    cv_folds=cv_folds,
                    positive_label=positive_label,
                    overfit_gap_threshold=overfit_gap_threshold,
                    near_perfect_accuracy=near_perfect_accuracy,
                ))
            if completed % max(1, len(seeds) * 2) == 0:
                print(f"Selesai {completed}/{total} konfigurasi model-skenario-seed")
    return pd.DataFrame(rows)


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Merangkum rata-rata dan simpangan baku metrik lintas seed."""
    metrics = [
        "train_accuracy", "test_accuracy", "train_precision", "test_precision",
        "train_recall", "test_recall", "train_f1", "test_f1",
        "train_roc_auc", "test_roc_auc", "cv_accuracy_mean",
        "cv_precision_macro_mean", "cv_recall_macro_mean", "cv_f1_macro_mean",
    ]
    summary = results.groupby(
        ["scenario", "missing_strategy", "scaler", "categorical_encoder", "algorithm"],
        dropna=False,
    )[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()
