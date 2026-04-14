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
    n_clusters: int


def _build_neighbor_graph(features: np.ndarray, eps: float, *, chunk_size: int = 128) -> list[np.ndarray]:
    """Compute epsilon-neighborhoods without materializing a full distance matrix."""
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError("Distance features must be a 2D matrix.")

    squared_norms = np.einsum("ij,ij->i", features, features)
    eps_squared = eps * eps
    neighbors: list[np.ndarray] = [np.empty(0, dtype=int) for _ in range(features.shape[0])]

    for start_index in range(0, features.shape[0], chunk_size):
        stop_index = min(start_index + chunk_size, features.shape[0])
        block = features[start_index:stop_index]
        block_squared_norms = np.einsum("ij,ij->i", block, block)[:, None]

        squared_distances = block_squared_norms + squared_norms[None, :] - 2.0 * (block @ features.T)
        np.maximum(squared_distances, 0.0, out=squared_distances)
        within_eps = squared_distances <= eps_squared + 1e-12

        for row_offset, mask in enumerate(within_eps):
            neighbors[start_index + row_offset] = np.flatnonzero(mask)

    return neighbors


def _expand_cluster(
    *,
    seed_index: int,
    seed_neighbors: Sequence[int],
    cluster_id: int,
    data: np.ndarray,
    labels: np.ndarray,
    core_mask: np.ndarray,
    neighbor_graph: Sequence[np.ndarray],
    min_pts: int,
    importance_rule: ImportanceRule,
) -> None:
    """Grow one cluster using the DBSCAN queue expansion idea."""
    labels[seed_index] = cluster_id
    core_mask[seed_index] = True

    queue = deque(int(index) for index in seed_neighbors)
    queued = set(int(index) for index in seed_neighbors)

    while queue:
        point_index = queue.popleft()

        if labels[point_index] == NOISE:
            labels[point_index] = cluster_id

        if labels[point_index] != UNVISITED:
            continue

        labels[point_index] = cluster_id
        current_neighbors = neighbor_graph[point_index]
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
    require_seed_importance: bool,
) -> ClusteringResult:
    """Shared implementation for DBSCAN and SS-DBSCAN."""
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if min_pts < 2:
        raise ValueError("min_pts must be at least 2.")

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D matrix.")

    distance_columns = tuple(distance_columns)
    distance_features = data[:, distance_columns]
    neighbor_graph = _build_neighbor_graph(distance_features, eps)

    labels = np.full(data.shape[0], UNVISITED, dtype=int)
    core_mask = np.zeros(data.shape[0], dtype=bool)
    cluster_id = 0

    for point_index in range(data.shape[0]):
        if labels[point_index] != UNVISITED:
            continue

        neighbors = neighbor_graph[point_index]
        seed_is_core = len(neighbors) >= min_pts and (
            (not require_seed_importance) or importance_rule(data[point_index], point_index)
        )
        if not seed_is_core:
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
            neighbor_graph=neighbor_graph,
            min_pts=min_pts,
            importance_rule=importance_rule,
        )

    return ClusteringResult(
        labels=labels,
        core_mask=core_mask,
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
        require_seed_importance=True,
    )


def ss_dbscan(
    data: np.ndarray,
    *,
    eps: float,
    min_pts: int,
    importance_rule: ImportanceRule,
    distance_columns: Iterable[int] = (0, 1),
    require_seed_importance: bool = False,
) -> ClusteringResult:
    """Run SS-DBSCAN with a user-defined core-point importance rule."""
    return _density_clustering(
        data,
        eps=eps,
        min_pts=min_pts,
        distance_columns=distance_columns,
        importance_rule=importance_rule,
        require_seed_importance=require_seed_importance,
    )
