"""Dataset loading and synthetic generation helpers for SS-DBSCAN experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

LETTERS_IMPORTANCE_COLUMNS = (5, 6, 7, 8, 10, 11, 12, 13, 14, 15)
LETTERS_IMPORTANCE_MARGIN = 2.0

IRIS_IMPORTANCE_COLUMNS = (2, 3)
IRIS_IMPORTANCE_LOWER_QUANTILE = 0.15
IRIS_IMPORTANCE_UPPER_QUANTILE = 0.85

SYNTHETIC_IMPORTANCE_RADIUS_COLUMN = 2
SYNTHETIC_IMPORTANCE_MULTIPLIER = 2.0


@dataclass(frozen=True)
class ImportanceProfile:
    """Dataset-specific information used by the SS-DBSCAN importance rule."""

    important_mask: np.ndarray
    scores: np.ndarray
    description: str


@dataclass(frozen=True)
class ExperimentDataset:
    """Dataset plus metadata needed by the project pipeline."""

    dataset_id: str
    display_name: str
    points: np.ndarray
    true_labels: np.ndarray
    true_label_display: np.ndarray
    feature_names: tuple[str, ...]
    visualization_points: np.ndarray
    visualization_axis_labels: tuple[str, str]
    importance_profile: ImportanceProfile
    distance_columns: tuple[int, ...]
    source_description: str


@dataclass(frozen=True)
class SyntheticDataset:
    """A reproducible dataset with two dense clusters, a bridge, and noise."""

    points: np.ndarray
    true_labels: np.ndarray
    feature_names: tuple[str, ...] = ("x", "y", "importance_radius")


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

    A point is marked important when at least one selected feature is within
    `margin` of that feature's maximum observed value.
    """
    points = np.asarray(points, dtype=float)
    selected_values = points[:, selected_columns]
    thresholds = selected_values.max(axis=0) - margin
    near_max = selected_values >= thresholds
    scores = near_max.sum(axis=1).astype(float)

    selected_feature_names = ", ".join(str(column + 1) for column in selected_columns)
    description = (
        f"any of features {selected_feature_names} within {margin:.0f} of that feature's max"
    )

    return ImportanceProfile(
        important_mask=scores > 0,
        scores=scores,
        description=description,
    )


def compute_iris_importance_profile(
    points: np.ndarray,
    *,
    selected_columns: tuple[int, ...] = IRIS_IMPORTANCE_COLUMNS,
    lower_quantile: float = IRIS_IMPORTANCE_LOWER_QUANTILE,
    upper_quantile: float = IRIS_IMPORTANCE_UPPER_QUANTILE,
) -> ImportanceProfile:
    """
    Mark iris points as important when both petal features stay in the central band.

    This encourages SS-DBSCAN to expand through representative petal shapes
    rather than relying on every dense boundary point.
    """
    points = np.asarray(points, dtype=float)
    selected_values = points[:, selected_columns]
    lower_bounds = np.quantile(selected_values, lower_quantile, axis=0)
    upper_bounds = np.quantile(selected_values, upper_quantile, axis=0)
    within_band = (selected_values >= lower_bounds) & (selected_values <= upper_bounds)
    scores = within_band.sum(axis=1).astype(float)

    selected_feature_names = ", ".join(str(column + 1) for column in selected_columns)
    description = (
        f"features {selected_feature_names} inside the {int(lower_quantile * 100)}th-"
        f"{int(upper_quantile * 100)}th percentile band"
    )

    return ImportanceProfile(
        important_mask=within_band.all(axis=1),
        scores=scores,
        description=description,
    )


def compute_synthetic_importance_profile(
    points: np.ndarray,
    *,
    radius_column: int = SYNTHETIC_IMPORTANCE_RADIUS_COLUMN,
    multiplier: float = SYNTHETIC_IMPORTANCE_MULTIPLIER,
) -> ImportanceProfile:
    """Use the provided paper-style synthetic radius rule."""
    points = np.asarray(points, dtype=float)
    radii = points[:, radius_column]
    threshold = float(multiplier * radii.mean())

    return ImportanceProfile(
        important_mask=radii > threshold,
        scores=radii.astype(float),
        description=f"importance_radius > {multiplier:.1f} * mean(importance_radius)",
    )


def load_letters_dataset(csv_path: str | Path) -> ExperimentDataset:
    """Load the letters dataset where the final column contains the class label."""
    csv_path = Path(csv_path)
    raw = np.loadtxt(csv_path, delimiter=",", dtype=float)

    if raw.ndim != 2 or raw.shape[1] < 2:
        raise ValueError("lettersPreProc.csv must contain feature columns and a label column.")

    points = raw[:, :-1]
    true_labels = raw[:, -1].astype(int)
    importance_profile = compute_letters_importance_profile(points)

    return ExperimentDataset(
        dataset_id="letters",
        display_name="lettersPreProc.csv",
        points=points,
        true_labels=true_labels,
        true_label_display=true_labels.astype(str),
        feature_names=tuple(f"feature_{index}" for index in range(1, points.shape[1] + 1)),
        visualization_points=_project_to_2d(points),
        visualization_axis_labels=("Principal component 1", "Principal component 2"),
        importance_profile=importance_profile,
        distance_columns=tuple(range(points.shape[1])),
        source_description=str(csv_path),
    )


def load_iris_dataset(arff_path: str | Path) -> ExperimentDataset:
    """Load the Weka iris dataset from an ARFF file."""
    arff_path = Path(arff_path)
    attribute_names: list[str] = []
    class_values: list[str] | None = None
    data_rows: list[list[str]] = []
    in_data_section = False

    for raw_line in arff_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue

        lower_line = line.lower()
        if not in_data_section:
            if lower_line.startswith("@attribute"):
                definition = line[len("@attribute") :].strip()
                attribute_name, attribute_type = definition.split(None, 1)
                attribute_name = attribute_name.strip("\"'")
                attribute_names.append(attribute_name)

                if attribute_type.startswith("{") and attribute_type.endswith("}"):
                    class_values = [value.strip() for value in attribute_type[1:-1].split(",")]
            elif lower_line.startswith("@data"):
                in_data_section = True
        else:
            data_rows.append([value.strip() for value in line.split(",")])

    if not data_rows or class_values is None:
        raise ValueError("dataset/iris.arff could not be parsed as a labeled ARFF dataset.")

    feature_names = tuple(attribute_names[:-1])
    points = np.array([[float(value) for value in row[:-1]] for row in data_rows], dtype=float)
    class_names = [row[-1] for row in data_rows]
    class_to_index = {class_name: index for index, class_name in enumerate(class_values)}
    true_labels = np.array([class_to_index[class_name] for class_name in class_names], dtype=int)
    importance_profile = compute_iris_importance_profile(points)

    return ExperimentDataset(
        dataset_id="iris",
        display_name="dataset/iris.arff",
        points=points,
        true_labels=true_labels,
        true_label_display=np.array(class_names, dtype=object),
        feature_names=feature_names,
        visualization_points=_project_to_2d(points),
        visualization_axis_labels=("Principal component 1", "Principal component 2"),
        importance_profile=importance_profile,
        distance_columns=tuple(range(points.shape[1])),
        source_description=str(arff_path),
    )


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


def make_synthetic_experiment_dataset() -> ExperimentDataset:
    """Create the synthetic varied-density dataset used for the third experiment."""
    synthetic = make_varied_density_dataset()
    importance_profile = compute_synthetic_importance_profile(synthetic.points)

    return ExperimentDataset(
        dataset_id="synthetic",
        display_name="synthetic_varied_density.csv",
        points=synthetic.points,
        true_labels=synthetic.true_labels,
        true_label_display=synthetic.true_labels.astype(str),
        feature_names=synthetic.feature_names,
        visualization_points=synthetic.points[:, :2],
        visualization_axis_labels=("x", "y"),
        importance_profile=importance_profile,
        distance_columns=(0, 1),
        source_description="generated in run_project.py",
    )


def save_dataset_csv(dataset: ExperimentDataset, csv_path: str | Path) -> None:
    """Save an experiment dataset to a CSV file with the label as the final column."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack((dataset.points, dataset.true_labels))
    np.savetxt(csv_path, rows, delimiter=",", fmt="%.8f")
