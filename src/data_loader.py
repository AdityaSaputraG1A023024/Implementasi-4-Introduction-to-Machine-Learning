"""Memuat German Credit dari OpenML atau data tabular sintetik sebagai fallback."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification


@dataclass
class DatasetBundle:
    """Data, target, dan catatan sumber untuk satu eksperimen."""

    features: pd.DataFrame
    target: pd.Series
    metadata: dict[str, Any]


def _synthetic_dataset(config: dict[str, Any]) -> DatasetBundle:
    """Membangun fitur numerik/kategorikal dengan kelas minoritas terkontrol."""
    settings = config["dataset"]
    features, labels = make_classification(
        n_samples=int(settings["fallback_samples"]),
        n_features=int(settings["fallback_features"]),
        n_informative=int(settings["fallback_informative"]),
        n_redundant=int(settings["fallback_redundant"]),
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[1.0 - float(settings["minority_class_weight"]),
                 float(settings["minority_class_weight"])],
        class_sep=0.85,
        flip_y=0.04,
        random_state=int(settings["synthetic_seed"]),
    )
    frame = pd.DataFrame(
        features,
        columns=[f"numerik_{index:02d}" for index in range(features.shape[1])],
    )
    categories = np.array(["rendah", "sedang", "tinggi"])
    for index in range(3):
        values = frame[f"numerik_{index:02d}"].to_numpy()
        cuts = np.quantile(values, [1 / 3, 2 / 3])
        frame[f"kategori_{index + 1}"] = pd.cut(
            values,
            bins=[-np.inf, cuts[0], cuts[1], np.inf],
            labels=categories,
        ).astype("object")
    target = pd.Series(labels.astype(int), name="kelas")
    return DatasetBundle(
        features=frame,
        target=target,
        metadata={
            "dataset_name": "make_classification (data sintetis)",
            "dataset_source": "sklearn.datasets.make_classification",
            "source_used": "synthetic",
            "openml_id": None,
            "original_rows": int(len(frame)),
            "original_columns": int(frame.shape[1]),
            "original_missing_cells": int(frame.isna().sum().sum()),
            "class_counts": target.value_counts().sort_index().to_dict(),
            "categorical_columns": ["kategori_1", "kategori_2", "kategori_3"],
            "numeric_columns": [
                column for column in frame if column.startswith("numerik_")
            ],
            "synthetic_seed": int(settings["synthetic_seed"]),
            "minority_class_weight_requested": float(settings["minority_class_weight"]),
        },
    )


def _fetch_openml_dataset(config: dict[str, Any]) -> DatasetBundle:
    """Mengunduh German Credit dari OpenML dan menormalkan kolom target."""
    settings = config["dataset"]
    bunch = fetch_openml(
        name=str(settings["openml_name"]),
        version=int(settings["version"]),
        as_frame=True,
        parser="auto",
    )
    features = bunch.data.copy()
    target = pd.Series(bunch.target, name="kelas").reset_index(drop=True)
    features = features.reset_index(drop=True)
    labelled = target.notna()
    features = features.loc[labelled].reset_index(drop=True)
    target = target.loc[labelled].astype("object").astype(str).reset_index(drop=True)
    target = target.astype("category").cat.codes.astype(int).rename("kelas")
    categorical = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return DatasetBundle(
        features=features,
        target=target,
        metadata={
            "dataset_name": "German Credit (OpenML credit-g)",
            "dataset_source": "OpenML: https://www.openml.org/d/31",
            "source_used": "openml",
            "openml_id": 31,
            "original_rows": int(len(features)),
            "original_columns": int(features.shape[1]),
            "original_missing_cells": int(features.isna().sum().sum()),
            "class_counts": target.value_counts().sort_index().to_dict(),
            "categorical_columns": categorical,
            "numeric_columns": [column for column in features if column not in categorical],
            "synthetic_seed": None,
            "minority_class_weight_requested": None,
        },
    )


def load_dataset(config: dict[str, Any]) -> DatasetBundle:
    """Memuat sumber terpilih; sumber auto memakai fallback dengan peringatan."""
    source = str(config["dataset"].get("source", "auto"))
    if source == "synthetic":
        return _synthetic_dataset(config)
    if source not in {"auto", "openml"}:
        raise ValueError("dataset.source harus 'auto', 'openml', atau 'synthetic'.")
    try:
        return _fetch_openml_dataset(config)
    except Exception as error:
        if source == "openml":
            raise RuntimeError(
                "Dataset OpenML diminta secara eksplisit, tetapi tidak berhasil diunduh."
            ) from error
        warnings.warn(
            f"OpenML tidak dapat diakses ({type(error).__name__}: {error}); "
            "menggunakan data sintetis berseed tetap.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _synthetic_dataset(config)


def inject_missing_values(
    features: pd.DataFrame,
    rates: list[float],
    seed: int,
) -> dict[float, pd.DataFrame]:
    """Menyisipkan satu NaN pada fitur per baris terpilih, bukan pada target."""
    if any(rate < 0.0 or rate >= 1.0 for rate in rates):
        raise ValueError("Tingkat missing harus >= 0 dan < 1.")
    output: dict[float, pd.DataFrame] = {}
    rows, columns = features.shape
    if columns == 0:
        raise ValueError("Dataset tidak memiliki kolom fitur untuk penyisipan missing.")
    for rate_index, rate in enumerate(rates):
        damaged = features.copy(deep=True)
        affected_rows = int(round(rows * rate))
        if affected_rows:
            generator = np.random.default_rng(seed + rate_index)
            row_positions = generator.choice(rows, size=affected_rows, replace=False)
            column_positions = generator.integers(0, columns, size=affected_rows)
            for row_index, column_index in zip(row_positions, column_positions):
                damaged.iat[int(row_index), int(column_index)] = np.nan
        output[float(rate)] = damaged
    return output


def inject_duplicate_rows(features: pd.DataFrame, target: pd.Series, count: int, seed: int) -> tuple[pd.DataFrame, pd.Series, int]:
    """Menambahkan salinan baris berseed tetap untuk menguji pembersihan duplikat."""
    if count < 0:
        raise ValueError("Jumlah duplikat tidak boleh negatif.")
    if count == 0:
        return features.copy(), target.copy(), 0
    generator = np.random.default_rng(seed)
    selected = generator.choice(len(features), size=count, replace=count > len(features))
    duplicate_features = features.iloc[selected].copy().reset_index(drop=True)
    duplicate_target = target.iloc[selected].copy().reset_index(drop=True)
    combined_features = pd.concat([features.reset_index(drop=True), duplicate_features], ignore_index=True)
    combined_target = pd.concat([target.reset_index(drop=True), duplicate_target], ignore_index=True)
    return combined_features, combined_target, count


def class_proportions(target: pd.Series) -> dict[str, float]:
    """Menghitung proporsi tiap label untuk dokumentasi dataset."""
    proportions = target.value_counts(normalize=True).sort_index()
    return {str(label): float(value) for label, value in proportions.items()}
