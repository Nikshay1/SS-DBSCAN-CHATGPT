"""Plain DBSCAN and paper-inspired SS-DBSCAN implementations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

UNVISITED = 0
NOISE = -1

ImportanceRule = Callable[[np.ndarray, int], bool]


@dataclass(frozen=True)
class ClusteringResult:
    """Result container for a density-based clustering run."""

    labels: np.ndarray
    core_mask: np.ndarray
    distance_matrix: np.ndarray
    n_clusters: int


def _pairwise_distance_matrix(features: np.ndarray) -> np.ndarray:
    """Compute the full Euclidean distance matrix."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("Input data must be a 2D matrix.")

    deltas = features[:, np.newaxis, :] - features[np.newaxis, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def _neighbor_indices(distance_matrix: np.ndarray, point_index: int, eps: float) -> np.ndarray:
    """Return all points within eps, including the point itself."""
    return np.flatnonzero(distance_matrix[point_index] <= eps)


def _expand_cluster(
    *,
    seed_index: int,
    seed_neighbors: Sequence[int],
    cluster_id: int,
    data: np.ndarray,
    labels: np.ndarray,
    core_mask: np.ndarray,
    distance_matrix: np.ndarray,
    eps: float,
    min_pts: int,
    importance_rule: ImportanceRule,
) -> None:
    """Grow one cluster using the DBSCAN queue expansion idea."""
    labels[seed_index] = cluster_id
    core_mask[seed_index] = True

    queue = deque(int(idx) for idx in seed_neighbors)
    queued = set(int(idx) for idx in seed_neighbors)

    while queue:
        point_index = queue.popleft()

        if labels[point_index] == NOISE:
            labels[point_index] = cluster_id

        if labels[point_index] != UNVISITED:
            continue

        labels[point_index] = cluster_id
        current_neighbors = _neighbor_indices(distance_matrix, point_index, eps)
        is_core = len(current_neighbors) >= min_pts and importance_rule(
            data[point_index],
            point_index,
        )

        if not is_core:
            continue

        core_mask[point_index] = True
        for neighbor_index in current_neighbors:
            neighbor_index = int(neighbor_index)
            if neighbor_index not in queued:
                queue.append(neighbor_index)
                queued.add(neighbor_index)


def _density_clustering(
    data: np.ndarray,
    *,
    eps: float,
    min_pts: int,
    distance_columns: Iterable[int],
    importance_rule: ImportanceRule,
) -> ClusteringResult:
    """Shared implementation for DBSCAN and SS-DBSCAN."""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if min_pts < 2:
        raise ValueError("min_pts must be at least 2.")

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D matrix.")

    distance_features = data[:, tuple(distance_columns)]
    distance_matrix = _pairwise_distance_matrix(distance_features)

    labels = np.full(data.shape[0], UNVISITED, dtype=int)
    core_mask = np.zeros(data.shape[0], dtype=bool)
    cluster_id = 0

    for point_index in range(data.shape[0]):
        if labels[point_index] != UNVISITED:
            continue

        neighbors = _neighbor_indices(distance_matrix, point_index, eps)
        is_core = len(neighbors) >= min_pts and importance_rule(data[point_index], point_index)
        if not is_core:
            labels[point_index] = NOISE
            continue

        cluster_id += 1
        _expand_cluster(
            seed_index=point_index,
            seed_neighbors=neighbors,
            cluster_id=cluster_id,
            data=data,
            labels=labels,
            core_mask=core_mask,
            distance_matrix=distance_matrix,
            eps=eps,
            min_pts=min_pts,
            importance_rule=importance_rule,
        )

    return ClusteringResult(
        labels=labels,
        core_mask=core_mask,
        distance_matrix=distance_matrix,
        n_clusters=cluster_id,
    )


def dbscan(
    data: np.ndarray,
    *,
    eps: float,
    min_pts: int,
    distance_columns: Iterable[int] = (0, 1),
) -> ClusteringResult:
    """Run plain DBSCAN."""
    return _density_clustering(
        data,
        eps=eps,
        min_pts=min_pts,
        distance_columns=distance_columns,
        importance_rule=lambda _point, _index: True,
    )


def ss_dbscan(
    data: np.ndarray,
    *,
    eps: float,
    min_pts: int,
    importance_rule: ImportanceRule,
    distance_columns: Iterable[int] = (0, 1),
) -> ClusteringResult:
    """Run SS-DBSCAN with a user-defined core-point importance rule."""
    return _density_clustering(
        data,
        eps=eps,
        min_pts=min_pts,
        distance_columns=distance_columns,
        importance_rule=importance_rule,
    )
