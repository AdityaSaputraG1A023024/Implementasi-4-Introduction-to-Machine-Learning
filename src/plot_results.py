"""Membuat grafik perbandingan model dan skenario pra-pemrosesan."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_figures(results_path: Path, figures_dir: Path) -> None:
    """Menyimpan heatmap, confusion matrix, dan grafik metrik train/test."""
    results = pd.read_csv(results_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Heatmap menampilkan rata-rata F1 dan recall kelas minoritas untuk setiap model.
    for metric, filename, title in (
        ("test_f1", "heatmap_f1.png", "F1 kelas minoritas"),
        ("test_recall", "heatmap_recall.png", "Recall kelas minoritas"),
    ):
        table = results.pivot_table(
            index="algorithm", columns="scenario", values=metric, aggfunc="mean"
        )
        figure, axis = plt.subplots(figsize=(max(12, 1.1 * len(table.columns)), 6))
        sns.heatmap(table, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Skenario pra-pemrosesan dan tingkat missing")
        axis.set_ylabel("Algoritma")
        axis.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        figure.savefig(figures_dir / filename, dpi=160)
        plt.close(figure)

    # Confusion matrix terbaik dan terburuk dipilih dari F1 uji aktual.
    for label, row in (
        ("terbaik", results.loc[results["test_f1"].idxmax()]),
        ("terburuk", results.loc[results["test_f1"].idxmin()]),
    ):
        matrix = [
            [row["test_confusion_tn"], row["test_confusion_fp"]],
            [row["test_confusion_fn"], row["test_confusion_tp"]],
        ]
        figure, axis = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False,
            xticklabels=["Prediksi mayoritas", "Prediksi minoritas"],
            yticklabels=["Aktual mayoritas", "Aktual minoritas"], ax=axis,
        )
        axis.set_title(
            f"Confusion matrix {label}\n{row['algorithm']} | {row['scenario']} | seed {row['seed']}"
        )
        axis.set_xlabel("Prediksi")
        axis.set_ylabel("Kelas aktual")
        figure.tight_layout()
        figure.savefig(figures_dir / f"confusion_matrix_{label}.png", dpi=160)
        plt.close(figure)

    # Grafik ini memperlihatkan gap train-test untuk mendeteksi indikasi overfitting.
    metrics = ["accuracy", "precision", "recall", "f1"]
    means = results.groupby(["algorithm", "scenario"])[
        [f"{metric}_test" for metric in metrics]
        + [f"train_{metric}" for metric in metrics]
    ].mean().reset_index()
    long = means.melt(
        id_vars=["algorithm", "scenario"],
        var_name="metrik_data",
        value_name="nilai",
    )
    long[["split", "metric"]] = long["metrik_data"].str.split("_", n=1, expand=True)
    long["kombinasi"] = long["algorithm"] + " | " + long["scenario"]
    figure, axis = plt.subplots(figsize=(16, max(6, 0.3 * long["kombinasi"].nunique())))
    sns.scatterplot(
        data=long, x="kombinasi", y="nilai", hue="split", style="metric",
        s=65, alpha=0.8, ax=axis,
    )
    axis.set_ylim(-0.03, 1.03)
    axis.set_title("Perbandingan metrik data latih dan data uji (rata-rata seed)")
    axis.set_xlabel("Algoritma | skenario")
    axis.set_ylabel("Nilai metrik")
    axis.tick_params(axis="x", rotation=80)
    axis.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(figures_dir / "metrik_latih_vs_uji.png", dpi=150)
    plt.close(figure)
    print(f"Grafik disimpan ke: {figures_dir}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    create_figures(
        project_root / "reports" / "hasil_eksperimen.csv",
        project_root / "reports" / "figures",
    )
