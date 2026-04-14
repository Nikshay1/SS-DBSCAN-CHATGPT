"""Runtime benchmarking helpers for the letters dataset."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .algorithms import dbscan, ss_dbscan
from .datasets import compute_letters_importance_profile


@dataclass(frozen=True)
class RuntimeRow:
    n_points: int
    dbscan_seconds: float
    ss_dbscan_seconds: float


def make_letters_importance_rule(points: np.ndarray):
    """Build the reference letters-dataset importance rule."""
    importance_profile = compute_letters_importance_profile(points)

    def is_important(_point: np.ndarray, point_index: int) -> bool:
        return bool(importance_profile.important_mask[point_index])

    return importance_profile.description, is_important


def benchmark_runtime(
    points: np.ndarray,
    *,
    sample_sizes: tuple[int, ...] = (1000, 2000, 4000, 8000),
    eps: float,
    min_pts: int,
    repeats: int = 2,
    random_state: int = 42,
    distance_columns: tuple[int, ...] | None = None,
) -> list[RuntimeRow]:
    """Measure runtime of DBSCAN and SS-DBSCAN on random subsets of the letters dataset."""
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
            _description, importance_rule = make_letters_importance_rule(sample_points)

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
