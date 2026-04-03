"""Run the full DWDM lab project pipeline."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssdbscan_lab.algorithms import dbscan, ss_dbscan
from ssdbscan_lab.benchmark import benchmark_runtime, make_radius_importance_rule
from ssdbscan_lab.datasets import make_varied_density_dataset
from ssdbscan_lab.metrics import adjusted_rand_index, v_measure_score
from ssdbscan_lab.visualization import plot_cluster_comparison, plot_runtime_comparison

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"


def _cluster_count(labels):
    return len({int(label) for label in labels if int(label) != -1})


def _noise_count(labels):
    return int(sum(int(label) == -1 for label in labels))


def _write_metrics_summary(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "algorithm",
                "eps",
                "min_pts",
                "importance_threshold",
                "clusters_found",
                "noise_points",
                "core_points",
                "v_measure",
                "adjusted_rand_index",
                "runtime_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_runtime_summary(rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["n_points", "dbscan_seconds", "ss_dbscan_seconds"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "n_points": row.n_points,
                    "dbscan_seconds": f"{row.dbscan_seconds:.6f}",
                    "ss_dbscan_seconds": f"{row.ss_dbscan_seconds:.6f}",
                }
            )


def main() -> None:
    eps = 0.45
    min_pts = 5

    dataset = make_varied_density_dataset(
        n_per_cluster=170,
        n_bridge=70,
        n_noise=30,
        random_state=7,
    )

    importance_threshold, importance_rule = make_radius_importance_rule(
        dataset.points,
        radius_column=2,
        multiplier=2.0,
    )

    dbscan_start = perf_counter()
    dbscan_result = dbscan(
        dataset.points,
        eps=eps,
        min_pts=min_pts,
        distance_columns=(0, 1),
    )
    dbscan_runtime = perf_counter() - dbscan_start

    ss_dbscan_start = perf_counter()
    ss_dbscan_result = ss_dbscan(
        dataset.points,
        eps=eps,
        min_pts=min_pts,
        importance_rule=importance_rule,
        distance_columns=(0, 1),
    )
    ss_dbscan_runtime = perf_counter() - ss_dbscan_start

    metrics_rows = [
        {
            "algorithm": "DBSCAN",
            "eps": eps,
            "min_pts": min_pts,
            "importance_threshold": "not used",
            "clusters_found": _cluster_count(dbscan_result.labels),
            "noise_points": _noise_count(dbscan_result.labels),
            "core_points": int(dbscan_result.core_mask.sum()),
            "v_measure": f"{v_measure_score(dataset.true_labels, dbscan_result.labels):.4f}",
            "adjusted_rand_index": f"{adjusted_rand_index(dataset.true_labels, dbscan_result.labels):.4f}",
            "runtime_seconds": f"{dbscan_runtime:.6f}",
        },
        {
            "algorithm": "SS-DBSCAN",
            "eps": eps,
            "min_pts": min_pts,
            "importance_threshold": f"radius > {importance_threshold:.4f}",
            "clusters_found": _cluster_count(ss_dbscan_result.labels),
            "noise_points": _noise_count(ss_dbscan_result.labels),
            "core_points": int(ss_dbscan_result.core_mask.sum()),
            "v_measure": f"{v_measure_score(dataset.true_labels, ss_dbscan_result.labels):.4f}",
            "adjusted_rand_index": f"{adjusted_rand_index(dataset.true_labels, ss_dbscan_result.labels):.4f}",
            "runtime_seconds": f"{ss_dbscan_runtime:.6f}",
        },
    ]

    runtime_rows = benchmark_runtime(
        sample_sizes=(180, 300, 420, 540),
        eps=eps,
        min_pts=min_pts,
        repeats=3,
        random_state=19,
    )

    _write_metrics_summary(metrics_rows, OUTPUT_DIR / "metrics_summary.csv")
    _write_runtime_summary(runtime_rows, OUTPUT_DIR / "runtime_summary.csv")

    plot_cluster_comparison(
        dataset=dataset,
        dbscan_result=dbscan_result,
        ss_dbscan_result=ss_dbscan_result,
        importance_threshold=importance_threshold,
        output_path=FIGURE_DIR / "dbscan_vs_ssdbscan.png",
    )
    plot_runtime_comparison(
        rows=runtime_rows,
        output_path=FIGURE_DIR / "runtime_comparison.png",
    )

    print("Project run completed.")
    print(f"Metrics summary: {OUTPUT_DIR / 'metrics_summary.csv'}")
    print(f"Runtime summary: {OUTPUT_DIR / 'runtime_summary.csv'}")
    print(f"Cluster comparison graph: {FIGURE_DIR / 'dbscan_vs_ssdbscan.png'}")
    print(f"Runtime graph: {FIGURE_DIR / 'runtime_comparison.png'}")

    for row in metrics_rows:
        print(
            f"{row['algorithm']}: clusters={row['clusters_found']}, "
            f"noise={row['noise_points']}, V-measure={row['v_measure']}, "
            f"ARI={row['adjusted_rand_index']}, runtime={row['runtime_seconds']} sec"
        )


if __name__ == "__main__":
    main()
