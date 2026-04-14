"""Runtime benchmarking helpers for the project datasets."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from .algorithms import dbscan, ss_dbscan
from .datasets import (
    ImportanceProfile,
    compute_synthetic_importance_profile,
    make_varied_density_dataset,
)

ImportanceProfileBuilder = Callable[[np.ndarray], ImportanceProfile]


@dataclass(frozen=True)
class RuntimeRow:
    n_points: int
    dbscan_seconds: float
    ss_dbscan_seconds: float


def benchmark_dataset_subsets(
    points: np.ndarray,
    *,
    importance_profile_builder: ImportanceProfileBuilder,
    sample_sizes: tuple[int, ...],
    eps: float,
    min_pts: int,
    require_seed_importance: bool = False,
    repeats: int = 2,
    random_state: int = 42,
    distance_columns: tuple[int, ...] | None = None,
) -> list[RuntimeRow]:
    """Measure runtime of DBSCAN and SS-DBSCAN on random subsets of a fixed dataset."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("Benchmark points must be a 2D matrix.")

    total_points = points.shape[0]
    if distance_columns is None:
        distance_columns = tuple(range(points.shape[1]))

    rows: list[RuntimeRow] = []
    rng = np.random.default_rng(random_state)
    valid_sizes = [size for size in sample_sizes if size <= total_points]

    for n_points in valid_sizes:
        dbscan_times = []
        ss_dbscan_times = []

        for _repeat_index in range(repeats):
            sample_indices = np.sort(rng.choice(total_points, size=n_points, replace=False))
            sample_points = points[sample_indices]
            importance_profile = importance_profile_builder(sample_points)

            def importance_rule(_point: np.ndarray, point_index: int) -> bool:
                return bool(importance_profile.important_mask[point_index])

            start_time = perf_counter()
            dbscan(
                sample_points,
                eps=eps,
                min_pts=min_pts,
                distance_columns=distance_columns,
            )
            dbscan_times.append(perf_counter() - start_time)

            start_time = perf_counter()
            ss_dbscan(
                sample_points,
                eps=eps,
                min_pts=min_pts,
                importance_rule=importance_rule,
                distance_columns=distance_columns,
                require_seed_importance=require_seed_importance,
            )
            ss_dbscan_times.append(perf_counter() - start_time)

        rows.append(
            RuntimeRow(
                n_points=n_points,
                dbscan_seconds=sum(dbscan_times) / len(dbscan_times),
                ss_dbscan_seconds=sum(ss_dbscan_times) / len(ss_dbscan_times),
            )
        )

    return rows


def benchmark_synthetic_runtime(
    *,
    sample_sizes: tuple[int, ...] = (180, 300, 420, 540),
    eps: float,
    min_pts: int,
    repeats: int = 3,
    random_state: int = 42,
) -> list[RuntimeRow]:
    """Measure runtime on newly generated synthetic datasets of increasing size."""
    rows: list[RuntimeRow] = []

    for size_index, n_points in enumerate(sample_sizes):
        n_per_cluster = max(40, int(n_points * 0.39))
        n_bridge = max(20, int(n_points * 0.14))
        n_noise = max(10, n_points - (2 * n_per_cluster) - n_bridge)

        dbscan_times = []
        ss_dbscan_times = []

        for repeat_index in range(repeats):
            synthetic = make_varied_density_dataset(
                n_per_cluster=n_per_cluster,
                n_bridge=n_bridge,
                n_noise=n_noise,
                random_state=random_state + (size_index * 31) + repeat_index,
            )
            importance_profile = compute_synthetic_importance_profile(synthetic.points)

            def importance_rule(_point: np.ndarray, point_index: int) -> bool:
                return bool(importance_profile.important_mask[point_index])

            start_time = perf_counter()
            dbscan(
                synthetic.points,
                eps=eps,
                min_pts=min_pts,
                distance_columns=(0, 1),
            )
            dbscan_times.append(perf_counter() - start_time)

            start_time = perf_counter()
            ss_dbscan(
                synthetic.points,
                eps=eps,
                min_pts=min_pts,
                importance_rule=importance_rule,
                distance_columns=(0, 1),
                require_seed_importance=True,
            )
            ss_dbscan_times.append(perf_counter() - start_time)

        rows.append(
            RuntimeRow(
                n_points=synthetic.points.shape[0],
                dbscan_seconds=sum(dbscan_times) / len(dbscan_times),
                ss_dbscan_seconds=sum(ss_dbscan_times) / len(ss_dbscan_times),
            )
        )

    return rows
