"""Synthetic dataset generation for DBSCAN vs SS-DBSCAN comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticDataset:
    """A reproducible dataset with two dense clusters, a bridge, and noise."""

    points: np.ndarray
    true_labels: np.ndarray
    feature_names: tuple[str, ...] = ("x", "y", "importance_radius")


def _blob_radii(
    points: np.ndarray,
    *,
    center: np.ndarray,
    rng: np.random.Generator,
    important_fraction: float,
) -> np.ndarray:
    """Assign large radii to central points and small radii elsewhere."""
    radii = 1.0 + 0.15 * rng.random(points.shape[0])
    distance_from_center = np.linalg.norm(points - center, axis=1)
    important_count = max(10, int(points.shape[0] * important_fraction))
    important_indices = np.argsort(distance_from_center)[:important_count]
    radii[important_indices] = 8.0 + 0.35 * rng.standard_normal(important_count)
    return np.clip(radii, 0.25, None)


def make_varied_density_dataset(
    *,
    n_per_cluster: int = 170,
    n_bridge: int = 70,
    n_noise: int = 30,
    random_state: int = 42,
    important_fraction: float = 0.45,
) -> SyntheticDataset:
    """
    Create a toy dataset that highlights the weakness of plain DBSCAN.

    Plain DBSCAN tends to merge the two blobs through the bridge points.
    SS-DBSCAN can prevent that merge because bridge points are not important.
    """
    rng = np.random.default_rng(random_state)

    center_left = np.array([-2.0, 0.0])
    center_right = np.array([2.0, 0.0])

    left_blob = rng.normal(
        loc=center_left,
        scale=np.array([0.42, 0.27]),
        size=(n_per_cluster, 2),
    )
    right_blob = rng.normal(
        loc=center_right,
        scale=np.array([0.42, 0.27]),
        size=(n_per_cluster, 2),
    )

    bridge_x = np.linspace(-1.15, 1.15, n_bridge) + rng.normal(0.0, 0.05, n_bridge)
    bridge_y = rng.normal(0.0, 0.05, n_bridge)
    bridge = np.column_stack((bridge_x, bridge_y))

    noise = rng.uniform(low=[-4.2, -2.5], high=[4.2, 2.5], size=(n_noise, 2))

    left_radii = _blob_radii(
        left_blob,
        center=center_left,
        rng=rng,
        important_fraction=important_fraction,
    )
    right_radii = _blob_radii(
        right_blob,
        center=center_right,
        rng=rng,
        important_fraction=important_fraction,
    )
    bridge_radii = 0.45 + 0.08 * rng.random(n_bridge)
    noise_radii = 0.35 + 0.08 * rng.random(n_noise)

    coordinates = np.vstack((left_blob, right_blob, bridge, noise))
    radii = np.concatenate((left_radii, right_radii, bridge_radii, noise_radii))
    points = np.column_stack((coordinates, radii))

    true_labels = np.concatenate(
        (
            np.zeros(n_per_cluster, dtype=int),
            np.ones(n_per_cluster, dtype=int),
            np.full(n_bridge, -1, dtype=int),
            np.full(n_noise, -1, dtype=int),
        )
    )

    return SyntheticDataset(points=points, true_labels=true_labels)
