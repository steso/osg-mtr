from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pydicom


DICOM_TAGS = [
    "SeriesNumber",
    "SeriesDescription",
    "PatientPosition",
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturerModelName",
    "ImageOrientationPatient",
]
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


def acute_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.degrees(np.arccos(np.clip(abs(cosine), 0.0, 1.0))))


def first_series_header(
    patient_dir: Path, series_token: str
) -> tuple[Path, pydicom.Dataset]:
    dicom_dir = patient_dir / "0000"
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"Missing DICOM directory: {dicom_dir}")

    for path in sorted(dicom_dir.iterdir()):
        if not path.is_file():
            continue
        try:
            dataset = pydicom.dcmread(
                path, stop_before_pixels=True, specific_tags=DICOM_TAGS
            )
        except (OSError, pydicom.errors.InvalidDicomError):
            continue
        if series_token in str(getattr(dataset, "SeriesDescription", "")).lower():
            return path, dataset
    raise FileNotFoundError(f"No {series_token} DICOM series found under {dicom_dir}")


def series_geometry(source: Path, dataset: pydicom.Dataset) -> dict[str, np.ndarray]:
    orientation = np.asarray(dataset.ImageOrientationPatient, dtype=float)
    if orientation.shape != (6,):
        raise ValueError(f"Invalid ImageOrientationPatient in {source}")

    row_lps = orientation[:3]
    column_lps = orientation[3:]
    slice_lps = np.cross(row_lps, column_lps)
    row_ras = LPS_TO_RAS @ row_lps
    column_ras = LPS_TO_RAS @ column_lps
    slice_ras = LPS_TO_RAS @ slice_lps
    b0_ras = np.array([0.0, 0.0, 1.0])

    # dcm2niix stores these 3D sagittal series as [slice, -row, -column].
    # The map scripts preserve that array ordering without permutation.
    map_axes_ras = np.column_stack((slice_ras, -row_ras, -column_ras))
    b0_map = map_axes_ras.T @ b0_ras
    b0_map /= np.linalg.norm(b0_map)
    return {
        "row_lps": row_lps,
        "column_lps": column_lps,
        "slice_lps": slice_lps,
        "row_ras": row_ras,
        "column_ras": column_ras,
        "slice_ras": slice_ras,
        "b0_ras": b0_ras,
        "b0_map": b0_map,
    }


def orientation_record(patient_dir: Path) -> dict[str, object]:
    mtr_source, mtr_dataset = first_series_header(patient_dir, "mton")
    t2s_source, t2s_dataset = first_series_header(patient_dir, "t2star")
    mtr = series_geometry(mtr_source, mtr_dataset)
    t2s = series_geometry(t2s_source, t2s_dataset)

    record: dict[str, object] = {
        "patient": patient_dir.name,
        "patient_position": getattr(mtr_dataset, "PatientPosition", ""),
        "field_strength_t": getattr(mtr_dataset, "MagneticFieldStrength", ""),
        "manufacturer": getattr(mtr_dataset, "Manufacturer", ""),
        "model": getattr(mtr_dataset, "ManufacturerModelName", ""),
        "b0_basis": "B0=scanner Z; direction from source DICOM ImageOrientationPatient only",
        "map_axis_convention": "map(i,j,k)=[DICOM slice,-row,-column]",
        "mtr_series_number": getattr(mtr_dataset, "SeriesNumber", ""),
        "mtr_series_description": getattr(mtr_dataset, "SeriesDescription", ""),
        "mtr_source_dicom": str(mtr_source),
        "t2s_series_number": getattr(t2s_dataset, "SeriesNumber", ""),
        "t2s_series_description": getattr(t2s_dataset, "SeriesDescription", ""),
        "t2s_source_dicom": str(t2s_source),
    }
    record.update({f"b0_scanner_{axis}": value for axis, value in zip("xyz", (0.0, 0.0, 1.0))})
    for map_name, geometry in (("mtr", mtr), ("t2s", t2s)):
        for name in ("row_lps", "column_lps", "slice_lps"):
            record.update(
                {f"{map_name}_dicom_{name}_{axis}": float(value) for axis, value in zip("xyz", geometry[name])}
            )
        record.update(
            {f"b0_{map_name}_map_{axis}": float(value) for axis, value in zip("ijk", geometry["b0_map"])}
        )
    record.update(
        {
            "mtr_b0_to_row_deg": acute_angle_degrees(mtr["b0_ras"], mtr["row_ras"]),
            "mtr_b0_to_column_deg": acute_angle_degrees(mtr["b0_ras"], mtr["column_ras"]),
            "mtr_b0_to_slice_deg": acute_angle_degrees(mtr["b0_ras"], mtr["slice_ras"]),
            "t2s_b0_to_row_deg": acute_angle_degrees(t2s["b0_ras"], t2s["row_ras"]),
            "t2s_b0_to_column_deg": acute_angle_degrees(t2s["b0_ras"], t2s["column_ras"]),
            "t2s_b0_to_slice_deg": acute_angle_degrees(t2s["b0_ras"], t2s["slice_ras"]),
        }
    )
    return record


def export_orientations(root: Path, output: Path) -> tuple[int, list[str]]:
    records = []
    errors = []
    for patient_dir in sorted(root.glob("Patient_*")):
        try:
            records.append(orientation_record(patient_dir))
        except (FileNotFoundError, AttributeError, ValueError) as error:
            errors.append(f"{patient_dir.name}: {error}")
    if not records:
        raise RuntimeError("No patient orientation records could be extracted")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return len(records), errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Export B0/map-axis angles from source DICOM headers.")
    parser.add_argument("dicom_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dicom_b0_orientations.csv"))
    args = parser.parse_args()
    count, errors = export_orientations(args.dicom_root, args.output)
    print(f"Wrote {count} records to {args.output}")
    for error in errors:
        print(f"WARNING: {error}")


if __name__ == "__main__":
    main()
