"""Runtime benchmarking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .algorithms import dbscan, ss_dbscan
from .datasets import make_varied_density_dataset


@dataclass(frozen=True)
class RuntimeRow:
    n_points: int
    dbscan_seconds: float
    ss_dbscan_seconds: float


def make_radius_importance_rule(points, *, radius_column: int = 2, multiplier: float = 2.0):
    """Return the paper-style rule radius > multiplier * mean(radius)."""
    threshold = float(multiplier * points[:, radius_column].mean())

    def is_important(point, _index):
        return float(point[radius_column]) > threshold

    return threshold, is_important


def benchmark_runtime(
    *,
    sample_sizes: tuple[int, ...] = (180, 300, 420, 540),
    eps: float,
    min_pts: int,
    repeats: int = 3,
    random_state: int = 42,
) -> list[RuntimeRow]:
    """Measure runtime of DBSCAN and SS-DBSCAN on increasing dataset sizes."""
    rows: list[RuntimeRow] = []

    for size_index, n_points in enumerate(sample_sizes):
        n_per_cluster = max(40, int(n_points * 0.39))
        n_bridge = max(20, int(n_points * 0.14))
        n_noise = max(10, n_points - (2 * n_per_cluster) - n_bridge)

        dbscan_times = []
        ss_dbscan_times = []

        for repeat_index in range(repeats):
            dataset = make_varied_density_dataset(
                n_per_cluster=n_per_cluster,
                n_bridge=n_bridge,
                n_noise=n_noise,
                random_state=random_state + (size_index * 31) + repeat_index,
            )
            _threshold, importance_rule = make_radius_importance_rule(dataset.points)

            start_time = perf_counter()
            dbscan(dataset.points, eps=eps, min_pts=min_pts, distance_columns=(0, 1))
            dbscan_times.append(perf_counter() - start_time)

            start_time = perf_counter()
            ss_dbscan(
                dataset.points,
                eps=eps,
                min_pts=min_pts,
                importance_rule=importance_rule,
                distance_columns=(0, 1),
            )
            ss_dbscan_times.append(perf_counter() - start_time)

        rows.append(
            RuntimeRow(
                n_points=dataset.points.shape[0],
                dbscan_seconds=sum(dbscan_times) / len(dbscan_times),
                ss_dbscan_seconds=sum(ss_dbscan_times) / len(ss_dbscan_times),
            )
        )

    return rows
