"""Mengorkestrasi sumber data, skenario praproses, evaluasi, dan grafik."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import yaml

from src.data_loader import (
    class_proportions,
    inject_duplicate_rows,
    inject_missing_values,
    load_dataset,
)
from src.preprocessing import remove_duplicates
from src.plot_results import create_figures
from src.train_evaluate import evaluate_experiment, summarize_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict[str, Any]:
    """Membaca konfigurasi eksperimen dari YAML."""
    with path.open("r", encoding="utf-8-sig") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("config.yaml harus berisi objek YAML.")
    return config


def _expand_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Menggabungkan skenario praproses dengan tiap tingkat missing value."""
    expanded: list[dict[str, Any]] = []
    rates = [float(value) for value in config["dataset"]["missing_rates"]]
    for scenario in config["preprocessing_scenarios"]:
        for rate in rates:
            expanded.append({
                **scenario,
                "missing_rate": rate,
                "name": f"{scenario['name']}_missing_{int(rate * 100)}pct",
            })
    return expanded


def _parse_args() -> argparse.Namespace:
    """Menerima jalur config serta opsi sumber data dan jumlah seed."""
    parser = argparse.ArgumentParser(description="Eksperimen prapemrosesan klasifikasi.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--source", choices=("auto", "openml", "synthetic"), default=None)
    parser.add_argument("--seeds", type=int, default=None,
                        help="Batasi jumlah seed pertama untuk uji cepat (minimal 1).")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args()


def _validation_features(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Memeriksa duplikat dengan versi fitur NaN sebelum penyisipan missing acak."""
    return remove_duplicates(features, target)


def run(config: dict[str, Any], source_override: str | None, output_dir: Path,
        seed_limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Menjalankan seluruh percobaan dan mengembalikan laporan eksperimen."""
    bundle = load_dataset(config if source_override is None else {
        **config, "dataset": {**config["dataset"], "source": source_override}
    })
    features, target, _ = inject_duplicate_rows(
        bundle.features, bundle.target,
        int(config["dataset"]["duplicate_rows_to_add"]),
        int(config["dataset"]["synthetic_seed"]) + 1,
    )
    duplicate_count_added = int(config["dataset"]["duplicate_rows_to_add"])
    clean_features, clean_target, duplicate_summary = _validation_features(features, target)
    missing_versions = inject_missing_values(
        clean_features,
        [float(value) for value in config["dataset"]["missing_rates"]],
        int(config["dataset"]["missing_seed"]),
    )

    scenarios = _expand_scenarios(config)
    algorithms = config["experiment"]["algorithms"]
    seeds = [int(value) for value in config["experiment"]["random_seeds"]]
    if seed_limit is not None:
        if seed_limit < 1:
            raise ValueError("--seeds harus minimal satu.")
        seeds = seeds[:seed_limit]
    rows: list[pd.DataFrame] = []
    for rate, damaged_features in missing_versions.items():
        applicable = [item for item in scenarios if item["missing_rate"] == rate]
        inserted_mask = damaged_features.isna() & ~clean_features.isna()
        missing_cells = int(inserted_mask.sum().sum())
        missing_rows = int(inserted_mask.any(axis=1).sum())
        for scenario in applicable:
            scenario_meta = {key: scenario[key] for key in ("name", "missing", "scaler", "categorical_encoder")}
            evaluated = evaluate_experiment(
                features=damaged_features,
                target=clean_target,
                scenarios=[scenario_meta],
                algorithms=algorithms,
                seeds=seeds,
                test_size=float(config["experiment"]["test_size"]),
                cv_folds=int(config["experiment"]["cv_folds"]),
                overfit_gap_threshold=float(config["experiment"]["overfit_gap_threshold"]),
                near_perfect_accuracy=float(config["experiment"]["near_perfect_accuracy"]),
            )
            evaluated["missing_rate"] = rate
            evaluated["missing_cells_inserted"] = missing_cells
            evaluated["missing_rows_affected"] = missing_rows
            evaluated["duplicate_rows_added"] = duplicate_count_added
            evaluated["duplicates_before_removal"] = duplicate_summary["duplicates_before_removal"]
            evaluated["duplicates_after_removal"] = duplicate_summary["duplicates_after_removal"]
            evaluated["rows_after_duplicate_removal"] = len(clean_features)
            evaluated["dataset_name"] = bundle.metadata["dataset_name"]
            evaluated["dataset_source"] = bundle.metadata["dataset_source"]
            evaluated["source_used"] = bundle.metadata["source_used"]
            evaluated["dataset_rows_original"] = bundle.metadata["original_rows"]
            evaluated["dataset_columns_original"] = bundle.metadata["original_columns"]
            evaluated["class_proportions"] = json.dumps(class_proportions(clean_target), ensure_ascii=False)
            evaluated["python_version"] = platform.python_version()
            evaluated["sklearn_version"] = sklearn.__version__
            evaluated["pandas_version"] = pd.__version__
            evaluated["numpy_version"] = np.__version__
            rows.append(evaluated)
            print(f"Selesai skenario {scenario['name']} ({len(rows)}/{len(scenarios)})")

    results = pd.concat(rows, ignore_index=True)
    summary = summarize_results(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "hasil_eksperimen.csv"
    summary_path = output_dir / "ringkasan.csv"
    results.to_csv(results_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    figures_dir = output_dir / "figures"
    create_figures(results_path, figures_dir)
    metadata = {
        **bundle.metadata,
        "class_proportions_final": class_proportions(clean_target),
        "duplicate_rows_added": duplicate_count_added,
        **duplicate_summary,
        "rows_after_duplicate_removal": len(clean_features),
        "missing_cells_by_rate": {
            f"{rate:.0%}": int((frame.isna() & ~clean_features.isna()).sum().sum())
            for rate, frame in missing_versions.items()
        },
        "missing_rows_affected_by_rate": {
            f"{rate:.0%}": int(
                (frame.isna() & ~clean_features.isna()).any(axis=1).sum()
            )
            for rate, frame in missing_versions.items()
        },
        "random_seeds_used": seeds,
        "scenario_count": len(scenarios),
        "algorithm_count_including_baseline": len(algorithms) + 1,
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    return results, summary, metadata


def main() -> None:
    """Menjalankan eksperimen dan mencetak ringkasan hasil aktual."""
    arguments = _parse_args()
    config = _load_config(arguments.config)
    results, summary, metadata = run(
        config,
        arguments.source,
        arguments.output_dir,
        arguments.seeds,
    )
    metadata_path = arguments.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Metadata dataset:")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("Ringkasan F1/recall uji terbaik:")
    best = summary.sort_values("test_f1_mean", ascending=False).head(10)
    print(best[["scenario", "algorithm", "test_f1_mean", "test_f1_std",
                "test_recall_mean", "test_accuracy_mean"]].to_string(index=False))
    print(f"Hasil mentah: {arguments.output_dir / 'hasil_eksperimen.csv'}")
    print(f"Ringkasan: {arguments.output_dir / 'ringkasan.csv'}")


if __name__ == "__main__":
    main()
