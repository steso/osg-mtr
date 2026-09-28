from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def _remove_small_components(mask: np.ndarray, min_component_voxels: int) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 3)
    labels, count = ndimage.label(mask, structure=structure)
    if count == 0:
        raise ValueError("The segmentation contains no foreground voxels")
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_component_voxels
    keep[0] = False
    cleaned = keep[labels]
    if not cleaned.any():
        raise ValueError(
            f"The segmentation has no component with at least {min_component_voxels} voxels"
        )
    return cleaned


def _grow_mask_by_mtr(
    mask: np.ndarray,
    mtr: np.ndarray,
    spacing: tuple[float, ...],
    forbidden_mask: np.ndarray | None,
    max_growth_mm: float,
    threshold_percentile: float,
) -> tuple[np.ndarray, float]:
    inside_values = mtr[mask & np.isfinite(mtr)]
    if not len(inside_values):
        return mask, float("nan")
    threshold = float(np.percentile(inside_values, threshold_percentile))
    distance = ndimage.distance_transform_edt(~mask, sampling=spacing)
    eligible = np.isfinite(mtr) & (mtr >= threshold) & (distance <= max_growth_mm)
    if forbidden_mask is not None and forbidden_mask.any():
        distance_to_other = ndimage.distance_transform_edt(
            ~forbidden_mask, sampling=spacing
        )
        eligible &= ~forbidden_mask & (distance < distance_to_other)
    grown = ndimage.binary_propagation(
        mask,
        structure=ndimage.generate_binary_structure(3, 3),
        mask=mask | eligible,
    )
    return grown, threshold


def _world_points(indices: np.ndarray, affine: np.ndarray) -> np.ndarray:
    return nib.affines.apply_affine(affine, indices)


def _primary_centerline_path(
    indices: np.ndarray, world_points: np.ndarray
) -> tuple[np.ndarray, int, int]:
    index_lookup = {tuple(index): position for position, index in enumerate(indices)}
    rows = []
    columns = []
    weights = []
    for position, index in enumerate(indices):
        for offset in np.ndindex(3, 3, 3):
            delta = np.asarray(offset) - 1
            if not np.any(delta) or tuple(delta) < (0, 0, 0):
                continue
            neighbor = index_lookup.get(tuple(index + delta))
            if neighbor is None:
                continue
            weight = float(np.linalg.norm(world_points[position] - world_points[neighbor]))
            rows.extend((position, neighbor))
            columns.extend((neighbor, position))
            weights.extend((weight, weight))

    node_count = len(indices)
    graph = coo_matrix((weights, (rows, columns)), shape=(node_count, node_count)).tocsr()
    degrees = np.diff(graph.indptr)
    endpoint_count = int(np.count_nonzero(degrees == 1))
    branch_count = int(np.count_nonzero(degrees > 2))
    component_count, component_labels = connected_components(graph, directed=False)

    best_distance = -1.0
    best_start = 0
    best_end = 0
    for component in range(component_count):
        component_nodes = np.flatnonzero(component_labels == component)
        endpoints = component_nodes[degrees[component_nodes] == 1]
        candidates = endpoints if len(endpoints) >= 2 else component_nodes
        if len(candidates) < 2:
            continue
        distances = dijkstra(graph, directed=False, indices=candidates)
        candidate_distances = distances[:, candidates]
        start_offset, end_offset = np.unravel_index(
            np.nanargmax(np.where(np.isfinite(candidate_distances), candidate_distances, -1.0)),
            candidate_distances.shape,
        )
        distance = float(candidate_distances[start_offset, end_offset])
        if distance > best_distance:
            best_distance = distance
            best_start = int(candidates[start_offset])
            best_end = int(candidates[end_offset])

    if best_distance < 0:
        return np.asarray([0], dtype=int), endpoint_count, branch_count

    _, predecessors = dijkstra(
        graph, directed=False, indices=best_start, return_predecessors=True
    )
    path = [best_end]
    while path[-1] != best_start:
        predecessor = int(predecessors[path[-1]])
        if predecessor < 0:
            raise ValueError("Unable to trace the primary centerline path")
        path.append(predecessor)
    path.reverse()
    path_indices = np.asarray(path, dtype=int)
    endpoint_delta = world_points[path_indices[-1]] - world_points[path_indices[0]]
    dominant_axis = int(np.argmax(np.abs(endpoint_delta)))
    if endpoint_delta[dominant_axis] < 0:
        path_indices = path_indices[::-1]
    return path_indices, endpoint_count, branch_count


def _centerline_profile(
    points: np.ndarray, angles: np.ndarray
) -> tuple[list[float], list[float], list[float]]:
    positions = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )
    length = float(positions.max())
    bin_count = min(100, max(2, int(np.ceil(length)) + 1))
    edges = np.linspace(0.0, length + np.finfo(float).eps, bin_count + 1)
    bins = np.minimum(np.digitize(positions, edges) - 1, bin_count - 1)
    profile_positions = []
    profile_angles = []
    for bin_index in range(bin_count):
        selected = bins == bin_index
        if selected.any():
            profile_positions.append(float(np.mean(positions[selected])))
            profile_angles.append(float(np.median(angles[selected])))
    profile_magic = (
        3.0 * np.cos(np.radians(np.asarray(profile_angles))) ** 2 - 1.0
    )
    return profile_positions, profile_angles, profile_magic.tolist()


def world_vector_in_image_axes(affine: np.ndarray, vector_ras: np.ndarray) -> np.ndarray:
    linear = np.asarray(affine[:3, :3], dtype=float)
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(spacing == 0):
        raise ValueError("NIfTI affine contains a zero-length image axis")
    direction = linear / spacing
    image_vector = np.linalg.solve(direction, np.asarray(vector_ras, dtype=float))
    norm = np.linalg.norm(image_vector)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("Cannot transform the scanner Z vector into image coordinates")
    return image_vector / norm


def analyze_segmentation(
    segmentation_path: Path,
    output_dir: Path,
    b0_ras: np.ndarray | None = None,
    neighborhood_mm: float = 5.0,
    b0_basis: str = "ras",
    min_component_voxels: int = 20,
    source_label: int | None = None,
    reference_data: np.ndarray | None = None,
    forbidden_mask: np.ndarray | None = None,
    max_growth_mm: float = 1.0,
    growth_threshold_percentile: float = 75.0,
) -> dict[str, object]:
    image = nib.load(str(segmentation_path))
    data = np.asarray(image.dataobj)
    finite = np.isfinite(data)
    input_mask = finite & (data == source_label) if source_label is not None else finite & (data > 0)
    mask = _remove_small_components(input_mask, min_component_voxels)
    cleaned_voxels = int(mask.sum())
    growth_threshold = float("nan")
    if reference_data is not None:
        if reference_data.shape != mask.shape:
            raise ValueError("The MTR reference and segmentation shapes do not match")
        mask, growth_threshold = _grow_mask_by_mtr(
            mask,
            np.asarray(reference_data),
            image.header.get_zooms()[:3],
            forbidden_mask,
            max_growth_mm,
            growth_threshold_percentile,
        )
    skeleton = skeletonize(mask)
    skeleton_indices = np.argwhere(skeleton)
    if len(skeleton_indices) < 3:
        raise ValueError("The segmentation is too small to estimate a centerline")

    if b0_basis == "image":
        skeleton_world = skeleton_indices * np.asarray(image.header.get_zooms()[:3])
    elif b0_basis == "ras":
        skeleton_world = _world_points(skeleton_indices, image.affine)
    else:
        raise ValueError("b0_basis must be 'image' or 'ras'")
    path_indices, endpoint_count, branch_count = _primary_centerline_path(
        skeleton_indices, skeleton_world
    )
    centerline_indices = skeleton_indices[path_indices]
    centerline_world = skeleton_world[path_indices]
    centerline = np.zeros(mask.shape, dtype=bool)
    centerline[tuple(centerline_indices.T)] = True
    tree = cKDTree(centerline_world)
    b0 = np.asarray([0.0, 0.0, 1.0] if b0_ras is None else b0_ras, dtype=float)
    b0 /= np.linalg.norm(b0)
    local_angles = np.empty(len(centerline_world), dtype=float)

    for index, point in enumerate(centerline_world):
        neighbors = tree.query_ball_point(point, neighborhood_mm)
        if len(neighbors) < 3:
            _, nearest = tree.query(point, k=min(5, len(centerline_world)))
            neighbors = np.atleast_1d(nearest).tolist()
        local_points = centerline_world[neighbors]
        centered = local_points - local_points.mean(axis=0)
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        tangent = right_vectors[0]
        local_angles[index] = np.degrees(np.arccos(np.clip(abs(tangent @ b0), 0.0, 1.0)))

    profile_positions, profile_angles, profile_magic = _centerline_profile(
        centerline_world, local_angles
    )

    mask_indices = np.argwhere(mask)
    if b0_basis == "image":
        mask_world = mask_indices * np.asarray(image.header.get_zooms()[:3])
    else:
        mask_world = _world_points(mask_indices, image.affine)
    _, nearest_centerline = tree.query(mask_world)
    voxel_angles = local_angles[nearest_centerline]
    angle_map = np.full(mask.shape, np.nan, dtype=np.float32)
    angle_map[tuple(mask_indices.T)] = voxel_angles.astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    angle_path = output_dir / "tendon_angle_degrees.nii.gz"
    centerline_path = output_dir / "tendon_centerline.nii.gz"
    summary_path = output_dir / "tendon_angle_summary.csv"
    angle_header = image.header.copy()
    angle_header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(angle_map, image.affine, angle_header), angle_path)
    nib.save(nib.Nifti1Image(centerline.astype(np.uint8), image.affine), centerline_path)

    summary: dict[str, object] = {
        "segmentation": segmentation_path.name,
        "source_label": source_label if source_label is not None else "all_positive",
        "input_foreground_voxels": int(input_mask.sum()),
        "foreground_voxels": int(mask.sum()),
        "excluded_small_component_voxels": int(input_mask.sum() - cleaned_voxels),
        "region_grown_voxels": int(mask.sum() - cleaned_voxels),
        "region_growth_max_mm": max_growth_mm if reference_data is not None else 0.0,
        "region_growth_mtr_threshold": growth_threshold,
        "centerline_voxels": int(centerline.sum()),
        "skeleton_endpoint_count": endpoint_count,
        "skeleton_branch_voxels": branch_count,
        "profile_qc": "review_branched_skeleton" if branch_count else "clear",
        "mean_angle_deg": float(np.mean(voxel_angles)),
        "median_angle_deg": float(np.median(voxel_angles)),
        "std_angle_deg": float(np.std(voxel_angles)),
        "min_angle_deg": float(np.min(voxel_angles)),
        "max_angle_deg": float(np.max(voxel_angles)),
        "p2_mean": float(np.mean((3.0 * np.cos(np.radians(voxel_angles)) ** 2) - 1.0)),
        "b0_basis": b0_basis,
        "b0_i_or_ras_x": float(b0[0]),
        "b0_j_or_ras_y": float(b0[1]),
        "b0_k_or_ras_z": float(b0[2]),
        "profile_position_mm": profile_positions,
        "profile_angle_deg": profile_angles,
        "profile_magic_deviation_deg": [
            abs(angle - 54.7356) for angle in profile_angles
        ],
        "profile_magic_parameter": profile_magic,
    }
    csv_summary = {key: value for key, value in summary.items() if not isinstance(value, list)}
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_summary))
        writer.writeheader()
        writer.writerow(csv_summary)
    return summary


def analyze_segmentations(
    segmentation_paths: list[Path],
    output_dir: Path,
    b0: np.ndarray | None = None,
    b0_basis: str = "image",
    reference_path: Path | None = None,
    segmentation_names: list[str] | None = None,
) -> dict[str, object]:
    if not segmentation_paths:
        raise ValueError("Choose at least one NIfTI segmentation")
    if segmentation_names is not None and len(segmentation_names) != len(
        segmentation_paths
    ):
        raise ValueError("Each segmentation path must have one display name")

    reference = nib.load(str(reference_path or segmentation_paths[0]))
    reference_data = np.asarray(reference.dataobj) if reference_path is not None else None
    combined_angles = np.full(reference.shape, np.nan, dtype=np.float32)
    combined_labels = np.zeros(reference.shape, dtype=np.uint16)
    combined_centerlines = np.zeros(reference.shape, dtype=np.uint16)
    summaries = []
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[int, Path, int, np.ndarray]] = []
    all_input_foreground = np.zeros(reference.shape, dtype=bool)
    for path_index, segmentation_path in enumerate(segmentation_paths):
        image = nib.load(str(segmentation_path))
        if image.shape != reference.shape or not np.allclose(
            image.affine, reference.affine, atol=1e-4, rtol=0.0
        ):
            raise ValueError(
                f"{segmentation_path.name} does not match the reference NIfTI grid"
            )

        data = np.asarray(image.dataobj)
        finite_positive = data[np.isfinite(data) & (data > 0)]
        source_labels = np.unique(finite_positive)
        if not len(source_labels):
            raise ValueError(f"{segmentation_path.name} contains no foreground labels")
        if np.any(source_labels != np.floor(source_labels)):
            raise ValueError(
                f"{segmentation_path.name} must use positive integer segmentation labels"
            )

        for source_label_value in source_labels.astype(int):
            input_mask = np.isfinite(data) & (data == source_label_value)
            if np.any(all_input_foreground & input_mask):
                raise ValueError(
                    f"{segmentation_path.name} label {source_label_value} overlaps another tendon segmentation"
                )
            all_input_foreground |= input_mask
            entries.append((path_index, segmentation_path, int(source_label_value), input_mask))

    for tendon_label, (path_index, segmentation_path, source_label_value, input_mask) in enumerate(entries, start=1):
        tendon_dir = output_dir / f"tendon_{tendon_label:02d}"
        summary = analyze_segmentation(
            segmentation_path,
            tendon_dir,
            b0,
            b0_basis=b0_basis,
            source_label=source_label_value,
            reference_data=reference_data,
            forbidden_mask=all_input_foreground & ~input_mask,
        )
        if segmentation_names is not None:
            summary["segmentation"] = segmentation_names[path_index]
        angle_map = np.asarray(
            nib.load(str(tendon_dir / "tendon_angle_degrees.nii.gz")).dataobj
        )
        centerline = np.asarray(
            nib.load(str(tendon_dir / "tendon_centerline.nii.gz")).dataobj
        ) > 0
        foreground = np.isfinite(angle_map)
        if np.any(np.isfinite(combined_angles) & foreground):
            raise ValueError(
                f"MTR-guided growth for {segmentation_path.name} label {source_label_value} overlaps another tendon"
            )
        combined_angles[foreground] = angle_map[foreground]
        combined_labels[foreground] = tendon_label
        combined_centerlines[centerline] = tendon_label
        summary["tendon_label"] = tendon_label
        summaries.append(summary)

    combined_magic = np.full(reference.shape, np.nan, dtype=np.float32)
    combined_magic_deviation = np.full(reference.shape, np.nan, dtype=np.float32)
    foreground = np.isfinite(combined_angles)
    combined_magic[foreground] = np.clip(
        3.0 * np.cos(np.radians(combined_angles[foreground])) ** 2 - 1.0,
        -1.0,
        2.0,
    )
    combined_magic_deviation[foreground] = np.abs(
        combined_angles[foreground] - 54.7356
    )

    float_header = reference.header.copy()
    float_header.set_data_dtype(np.float32)
    label_header = reference.header.copy()
    label_header.set_data_dtype(np.uint16)
    nib.save(
        nib.Nifti1Image(combined_angles, reference.affine, float_header),
        output_dir / "combined_tendon_angle_degrees.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(combined_magic, reference.affine, float_header),
        output_dir / "combined_magic_angle_parameter.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(combined_magic_deviation, reference.affine, float_header),
        output_dir / "combined_magic_angle_deviation_degrees.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(combined_labels, reference.affine, label_header),
        output_dir / "combined_tendon_labels.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(combined_centerlines, reference.affine, label_header),
        output_dir / "combined_tendon_centerlines.nii.gz",
    )
    with (output_dir / "tendon_angle_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        csv_rows = [
            {key: value for key, value in summary.items() if not isinstance(value, list)}
            for summary in summaries
        ]
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    combined_values = combined_angles[foreground]
    return {
        "tendon_count": len(summaries),
        "foreground_voxels": int(foreground.sum()),
        "centerline_voxels": int(np.count_nonzero(combined_centerlines)),
        "mean_angle_deg": float(np.mean(combined_values)),
        "median_angle_deg": float(np.median(combined_values)),
        "min_angle_deg": float(np.min(combined_values)),
        "max_angle_deg": float(np.max(combined_values)),
        "magic_parameter_mean": float(np.mean(combined_magic[foreground])),
        "magic_deviation_mean_deg": float(
            np.mean(combined_magic_deviation[foreground])
        ),
        "tendons": summaries,
    }
