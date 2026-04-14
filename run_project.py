"""Run the SS-DBSCAN project pipeline on three datasets."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ssdbscan_lab.algorithms import dbscan, ss_dbscan
from ssdbscan_lab.benchmark import benchmark_dataset_subsets, benchmark_synthetic_runtime
from ssdbscan_lab.datasets import (
    ExperimentDataset,
    compute_iris_importance_profile,
    compute_letters_importance_profile,
    load_iris_dataset,
    load_letters_dataset,
    make_synthetic_experiment_dataset,
    save_dataset_csv,
)
from ssdbscan_lab.metrics import adjusted_rand_index, v_measure_score
from ssdbscan_lab.visualization import plot_cluster_comparison, plot_runtime_comparison

LETTERS_DATASET_PATH = PROJECT_ROOT / "dataset" / "lettersPreProc.csv"
IRIS_DATASET_PATH = PROJECT_ROOT / "dataset" / "iris.arff"
SYNTHETIC_DATASET_PATH = PROJECT_ROOT / "dataset" / "synthetic_varied_density.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_id: str
    loader: Callable[[], ExperimentDataset]
    importance_profile_builder: Callable | None
    eps: float
    min_pts: int
    require_seed_importance: bool
    benchmark_sample_sizes: tuple[int, ...]
    benchmark_repeats: int
    benchmark_mode: str


def _cluster_count(labels):
    return len({int(label) for label in labels if int(label) != -1})


def _noise_count(labels):
    return int(sum(int(label) == -1 for label in labels))


def _importance_rule_from_dataset(dataset: ExperimentDataset):
    def is_important(_point, point_index):
        return bool(dataset.importance_profile.important_mask[point_index])

    return is_important


def _write_csv(rows, output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_output_dir(dataset_id: str) -> Path:
    return OUTPUT_DIR / dataset_id


def _run_experiment(config: ExperimentConfig):
    dataset = config.loader()
    importance_rule = _importance_rule_from_dataset(dataset)

    dbscan_start = perf_counter()
    dbscan_result = dbscan(
        dataset.points,
        eps=config.eps,
        min_pts=config.min_pts,
        distance_columns=dataset.distance_columns,
    )
    dbscan_runtime = perf_counter() - dbscan_start

    ss_dbscan_start = perf_counter()
    ss_dbscan_result = ss_dbscan(
        dataset.points,
        eps=config.eps,
        min_pts=config.min_pts,
        importance_rule=importance_rule,
        distance_columns=dataset.distance_columns,
        require_seed_importance=config.require_seed_importance,
    )
    ss_dbscan_runtime = perf_counter() - ss_dbscan_start

    metrics_rows = [
        {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.display_name,
            "dataset_source": dataset.source_description,
            "algorithm": "DBSCAN",
            "eps": config.eps,
            "min_pts": config.min_pts,
            "importance_rule": "not used",
            "rows": dataset.points.shape[0],
            "features": dataset.points.shape[1],
            "clusters_found": _cluster_count(dbscan_result.labels),
            "noise_points": _noise_count(dbscan_result.labels),
            "core_points": int(dbscan_result.core_mask.sum()),
            "v_measure": f"{v_measure_score(dataset.true_labels, dbscan_result.labels):.4f}",
            "adjusted_rand_index": f"{adjusted_rand_index(dataset.true_labels, dbscan_result.labels):.4f}",
            "runtime_seconds": f"{dbscan_runtime:.6f}",
        },
        {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.display_name,
            "dataset_source": dataset.source_description,
            "algorithm": "SS-DBSCAN",
            "eps": config.eps,
            "min_pts": config.min_pts,
            "importance_rule": dataset.importance_profile.description,
            "rows": dataset.points.shape[0],
            "features": dataset.points.shape[1],
            "clusters_found": _cluster_count(ss_dbscan_result.labels),
            "noise_points": _noise_count(ss_dbscan_result.labels),
            "core_points": int(ss_dbscan_result.core_mask.sum()),
            "v_measure": f"{v_measure_score(dataset.true_labels, ss_dbscan_result.labels):.4f}",
            "adjusted_rand_index": f"{adjusted_rand_index(dataset.true_labels, ss_dbscan_result.labels):.4f}",
            "runtime_seconds": f"{ss_dbscan_runtime:.6f}",
        },
    ]

    if config.benchmark_mode == "subset":
        runtime_objects = benchmark_dataset_subsets(
            dataset.points,
            importance_profile_builder=config.importance_profile_builder,
            sample_sizes=config.benchmark_sample_sizes,
            eps=config.eps,
            min_pts=config.min_pts,
            require_seed_importance=config.require_seed_importance,
            repeats=config.benchmark_repeats,
            random_state=19,
            distance_columns=dataset.distance_columns,
        )
    else:
        runtime_objects = benchmark_synthetic_runtime(
            sample_sizes=config.benchmark_sample_sizes,
            eps=config.eps,
            min_pts=config.min_pts,
            repeats=config.benchmark_repeats,
            random_state=19,
        )

    runtime_rows = [
        {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.display_name,
            "n_points": row.n_points,
            "dbscan_seconds": f"{row.dbscan_seconds:.6f}",
            "ss_dbscan_seconds": f"{row.ss_dbscan_seconds:.6f}",
        }
        for row in runtime_objects
    ]

    assignment_rows = []
    for row_index, (true_label, true_label_display, db_label, ss_label, is_important, score) in enumerate(
        zip(
            dataset.true_labels,
            dataset.true_label_display,
            dbscan_result.labels,
            ss_dbscan_result.labels,
            dataset.importance_profile.important_mask,
            dataset.importance_profile.scores,
        )
    ):
        assignment_rows.append(
            {
                "dataset_id": dataset.dataset_id,
                "dataset_name": dataset.display_name,
                "row_index": row_index,
                "true_label": int(true_label),
                "true_label_display": str(true_label_display),
                "dbscan_label": int(db_label),
                "ss_dbscan_label": int(ss_label),
                "important_point": bool(is_important),
                "importance_score": f"{float(score):.6f}",
            }
        )

    dataset_output_dir = _dataset_output_dir(dataset.dataset_id)
    _write_csv(
        metrics_rows,
        dataset_output_dir / "metrics_summary.csv",
        [
            "dataset_id",
            "dataset_name",
            "dataset_source",
            "algorithm",
            "eps",
            "min_pts",
            "importance_rule",
            "rows",
            "features",
            "clusters_found",
            "noise_points",
            "core_points",
            "v_measure",
            "adjusted_rand_index",
            "runtime_seconds",
        ],
    )
    _write_csv(
        runtime_rows,
        dataset_output_dir / "runtime_summary.csv",
        ["dataset_id", "dataset_name", "n_points", "dbscan_seconds", "ss_dbscan_seconds"],
    )
    _write_csv(
        assignment_rows,
        dataset_output_dir / "cluster_assignments.csv",
        [
            "dataset_id",
            "dataset_name",
            "row_index",
            "true_label",
            "true_label_display",
            "dbscan_label",
            "ss_dbscan_label",
            "important_point",
            "importance_score",
        ],
    )

    plot_cluster_comparison(
        dataset=dataset,
        dbscan_result=dbscan_result,
        ss_dbscan_result=ss_dbscan_result,
        output_path=FIGURE_DIR / f"{dataset.dataset_id}_dbscan_vs_ssdbscan.png",
    )
    plot_runtime_comparison(
        rows=runtime_objects,
        dataset_name=dataset.display_name,
        output_path=FIGURE_DIR / f"{dataset.dataset_id}_runtime_comparison.png",
    )

    return dataset, metrics_rows, runtime_rows, assignment_rows


def _load_synthetic_dataset():
    dataset = make_synthetic_experiment_dataset()
    save_dataset_csv(dataset, SYNTHETIC_DATASET_PATH)
    return replace(dataset, source_description=str(SYNTHETIC_DATASET_PATH))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    configs = [
        ExperimentConfig(
            dataset_id="letters",
            loader=lambda: load_letters_dataset(LETTERS_DATASET_PATH),
            importance_profile_builder=compute_letters_importance_profile,
            eps=8.0,
            min_pts=17,
            require_seed_importance=False,
            benchmark_sample_sizes=(1000, 2000, 4000, 8000),
            benchmark_repeats=2,
            benchmark_mode="subset",
        ),
        ExperimentConfig(
            dataset_id="iris",
            loader=lambda: load_iris_dataset(IRIS_DATASET_PATH),
            importance_profile_builder=compute_iris_importance_profile,
            eps=0.9,
            min_pts=4,
            require_seed_importance=False,
            benchmark_sample_sizes=(50, 75, 100, 150),
            benchmark_repeats=4,
            benchmark_mode="subset",
        ),
        ExperimentConfig(
            dataset_id="synthetic",
            loader=_load_synthetic_dataset,
            importance_profile_builder=None,
            eps=0.45,
            min_pts=5,
            require_seed_importance=True,
            benchmark_sample_sizes=(180, 300, 420, 540),
            benchmark_repeats=3,
            benchmark_mode="synthetic",
        ),
    ]

    all_metric_rows = []
    all_runtime_rows = []
    all_assignment_rows = []

    for config in configs:
        dataset, metric_rows, runtime_rows, assignment_rows = _run_experiment(config)
        all_metric_rows.extend(metric_rows)
        all_runtime_rows.extend(runtime_rows)
        all_assignment_rows.extend(assignment_rows)

        print(f"[{dataset.dataset_id}] Dataset: {dataset.source_description}")
        print(f"[{dataset.dataset_id}] Rows: {dataset.points.shape[0]}, features: {dataset.points.shape[1]}")
        print(f"[{dataset.dataset_id}] Per-dataset outputs: {_dataset_output_dir(dataset.dataset_id)}")
        print(
            f"[{dataset.dataset_id}] Cluster figure: "
            f"{FIGURE_DIR / f'{dataset.dataset_id}_dbscan_vs_ssdbscan.png'}"
        )
        print(
            f"[{dataset.dataset_id}] Runtime figure: "
            f"{FIGURE_DIR / f'{dataset.dataset_id}_runtime_comparison.png'}"
        )
        for row in metric_rows:
            print(
                f"[{dataset.dataset_id}] {row['algorithm']}: clusters={row['clusters_found']}, "
                f"noise={row['noise_points']}, V-measure={row['v_measure']}, "
                f"ARI={row['adjusted_rand_index']}, runtime={row['runtime_seconds']} sec"
            )

    _write_csv(
        all_metric_rows,
        OUTPUT_DIR / "metrics_summary.csv",
        [
            "dataset_id",
            "dataset_name",
            "dataset_source",
            "algorithm",
            "eps",
            "min_pts",
            "importance_rule",
            "rows",
            "features",
            "clusters_found",
            "noise_points",
            "core_points",
            "v_measure",
            "adjusted_rand_index",
            "runtime_seconds",
        ],
    )
    _write_csv(
        all_runtime_rows,
        OUTPUT_DIR / "runtime_summary.csv",
        ["dataset_id", "dataset_name", "n_points", "dbscan_seconds", "ss_dbscan_seconds"],
    )
    _write_csv(
        all_assignment_rows,
        OUTPUT_DIR / "cluster_assignments.csv",
        [
            "dataset_id",
            "dataset_name",
            "row_index",
            "true_label",
            "true_label_display",
            "dbscan_label",
            "ss_dbscan_label",
            "important_point",
            "importance_score",
        ],
    )

    print("Project run completed for all datasets.")
    print(f"Combined metrics summary: {OUTPUT_DIR / 'metrics_summary.csv'}")
    print(f"Combined runtime summary: {OUTPUT_DIR / 'runtime_summary.csv'}")
    print(f"Combined cluster assignments: {OUTPUT_DIR / 'cluster_assignments.csv'}")


if __name__ == "__main__":
    main()
