"""Dataset loading helpers for the letters SS-DBSCAN experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LETTERS_IMPORTANCE_COLUMNS = (5, 6, 7, 8, 10, 11, 12, 13, 14, 15)
LETTERS_IMPORTANCE_MARGIN = 2.0


@dataclass(frozen=True)
class ImportanceProfile:
    """Dataset-specific information used by the SS-DBSCAN importance rule."""

    important_mask: np.ndarray
    scores: np.ndarray
    selected_columns: tuple[int, ...]
    thresholds: np.ndarray
    description: str


@dataclass(frozen=True)
class LettersDataset:
    """Letter-recognition dataset plus metadata needed by the pipeline."""

    points: np.ndarray
    true_labels: np.ndarray
    feature_names: tuple[str, ...]
    visualization_points: np.ndarray
    visualization_axis_labels: tuple[str, str]
    importance_profile: ImportanceProfile
    distance_columns: tuple[int, ...]


def _project_to_2d(points: np.ndarray) -> np.ndarray:
    """Project high-dimensional features to two principal components."""
    centered = points - points.mean(axis=0, keepdims=True)
    _u, _s, right_singular_vectors = np.linalg.svd(centered, full_matrices=False)
    return centered @ right_singular_vectors[:2].T


def compute_letters_importance_profile(
    points: np.ndarray,
    *,
    selected_columns: tuple[int, ...] = LETTERS_IMPORTANCE_COLUMNS,
    margin: float = LETTERS_IMPORTANCE_MARGIN,
) -> ImportanceProfile:
    """
    Recreate the rule used in the reference repo for the letters dataset.

    A point is marked important when at least one of the selected feature values
    is within `margin` of that feature's maximum observed value.
    """
    points = np.asarray(points, dtype=float)
    selected_values = points[:, selected_columns]
    thresholds = selected_values.max(axis=0) - margin
    near_max = selected_values >= thresholds
    scores = near_max.sum(axis=1).astype(int)

    selected_feature_names = ", ".join(str(column + 1) for column in selected_columns)
    description = (
        f"any of features {selected_feature_names} within {margin:.0f} of that feature's max"
    )

    return ImportanceProfile(
        important_mask=scores > 0,
        scores=scores,
        selected_columns=selected_columns,
        thresholds=thresholds,
        description=description,
    )


def load_letters_dataset(csv_path: str | Path) -> LettersDataset:
    """Load the letters dataset where the final column contains the class label."""
    csv_path = Path(csv_path)
    raw = np.loadtxt(csv_path, delimiter=",", dtype=float)

    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("lettersPreProc.csv must contain feature columns and a label column.")

    points = raw[:, :-1]
    true_labels = raw[:, -1].astype(int)
    importance_profile = compute_letters_importance_profile(points)

    return LettersDataset(
        points=points,
        true_labels=true_labels,
        feature_names=tuple(f"feature_{index}" for index in range(1, points.shape[1] + 1)),
        visualization_points=_project_to_2d(points),
        visualization_axis_labels=("Principal component 1", "Principal component 2"),
        importance_profile=importance_profile,
        distance_columns=tuple(range(points.shape[1])),
    )
