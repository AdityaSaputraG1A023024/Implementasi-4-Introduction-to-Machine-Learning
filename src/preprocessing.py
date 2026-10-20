"""Membangun pipeline pra-pemrosesan yang hanya fit pada data latih."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler


def _imputer(strategy: str) -> Any:
    """Memilih imputasi median/modus atau KNN berdasarkan tipe fitur."""
    if strategy == "drop":
        # Baris bermissing sudah dibuang terpisah sebelum CV dan fit model.
        return SimpleImputer(strategy="median", keep_empty_features=True)
    if strategy == "median":
        return SimpleImputer(strategy="median", keep_empty_features=True)
    if strategy == "mode":
        return SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    if strategy == "knn":
        return KNNImputer(n_neighbors=5, weights="distance", keep_empty_features=True)
    if strategy == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        return IterativeImputer(max_iter=10, random_state=0, keep_empty_features=True)
    raise ValueError(f"Strategi missing value tidak dikenal: {strategy}.")


def build_preprocessor(
    features: pd.DataFrame,
    missing_strategy: str,
    scaler: str,
    categorical_encoder: str,
) -> ColumnTransformer:
    """Menyusun imputasi, skala, dan encoder sebagai satu ColumnTransformer."""
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features if column not in numeric_columns]
    numeric_steps: list[tuple[str, Any]] = [("imputer", _imputer(missing_strategy))]

    if scaler == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif scaler == "minmax":
        numeric_steps.append(("scaler", MinMaxScaler()))
    elif scaler != "none":
        raise ValueError(f"Scaler tidak dikenal: {scaler}.")

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))

    if categorical_columns:
        encoder = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
        if categorical_encoder == "onehot":
            encoder_pipeline = Pipeline([
                ("imputer", encoder),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        elif categorical_encoder == "ordinal":
            encoder_pipeline = Pipeline([
                ("imputer", encoder),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )),
            ])
        else:
            raise ValueError(f"Encoder kategorikal tidak dikenal: {categorical_encoder}.")
        transformers.append(("categorical", encoder_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("Dataset tidak memiliki kolom fitur yang dapat diproses.")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def remove_duplicates(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Menghapus baris fitur duplikat secara konsisten dengan labelnya."""
    aligned_features = features.reset_index(drop=True)
    aligned_target = target.reset_index(drop=True)
    # Baris dianggap duplikat hanya bila fitur dan labelnya sama persis.
    full_rows = pd.concat([aligned_features, aligned_target.rename("__target__")], axis=1)
    duplicate_mask = full_rows.duplicated(keep="first")
    duplicates_before = int(duplicate_mask.sum())
    keep_mask = ~duplicate_mask
    clean_features = aligned_features.loc[keep_mask].reset_index(drop=True)
    clean_target = aligned_target.loc[keep_mask].reset_index(drop=True)
    clean_rows = pd.concat([clean_features, clean_target.rename("__target__")], axis=1)
    duplicates_after = int(clean_rows.duplicated().sum())
    return clean_features, clean_target, {
        "duplicates_before_removal": duplicates_before,
        "duplicates_after_removal": duplicates_after,
        "rows_removed": duplicates_before,
    }


def split_drop_missing(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    test_features: pd.DataFrame,
    test_target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Membuang baris yang memiliki missing fitur dalam setiap partisi terpisah."""
    train_keep = ~train_features.isna().any(axis=1)
    test_keep = ~test_features.isna().any(axis=1)
    return (
        train_features.loc[train_keep],
        train_target.loc[train_keep],
        test_features.loc[test_keep],
        test_target.loc[test_keep],
    )

