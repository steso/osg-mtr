from __future__ import annotations

import argparse
import csv
import re
import tempfile
import uuid
import webbrowser
from pathlib import Path
from threading import Timer

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from tendon_geometry import analyze_segmentations


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
ORIENTATION_CSV = BASE_DIR / "dicom_b0_orientations.csv"
DEFAULT_SEGMENTATION_DIR = BASE_DIR / "UTE MT Revision"
DEFAULT_MTR_DIR = Path(r"Y:\_collab_raw\Raw Data UTE MT Foot\dicom_pacs\mtr")
app = Flask(__name__)


def patient_from_filename(filename: str) -> str:
    match = re.search(r"patient[_ -]?(\d+)", filename, re.IGNORECASE)
    if not match:
        raise ValueError("Filename must contain a patient number, for example Patient_27_mtr.nii.gz")
    return f"Patient_{int(match.group(1)):02d}"


def map_type_from_filename(filename: str) -> str:
    lowered = filename.lower()
    if "mtr" in lowered:
        return "mtr"
    if "t2s" in lowered or "t2star" in lowered or "t2*" in lowered:
        return "t2s"
    raise ValueError("Filename must identify the map type with 'mtr' or 't2s'")


def dicom_orientation_for_filename(filename: str) -> dict[str, object]:
    patient = patient_from_filename(filename)
    map_type = map_type_from_filename(filename)
    if not ORIENTATION_CSV.exists():
        raise ValueError("DICOM orientation CSV is missing; run dicom_orientation.py first")
    with ORIENTATION_CSV.open(newline="", encoding="utf-8") as stream:
        record = next((row for row in csv.DictReader(stream) if row["patient"] == patient), None)
    if record is None:
        raise ValueError(f"No DICOM orientation record found for {patient}")
    return {
        "patient": patient,
        "map_type": map_type,
        "b0_image_i": float(record[f"b0_{map_type}_map_i"]),
        "b0_image_j": float(record[f"b0_{map_type}_map_j"]),
        "b0_image_k": float(record[f"b0_{map_type}_map_k"]),
        "angulation_deg": float(record[f"{map_type}_b0_to_column_deg"]),
        "source_dicom": record[f"{map_type}_source_dicom"],
        "map_axis_convention": record["map_axis_convention"],
    }


def _unique_nifti_files(root: Path) -> list[Path]:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))
    )
    unique: dict[str, Path] = {}
    for path in sorted(files, key=lambda item: not item.name.lower().endswith(".nii.gz")):
        key = str(path.relative_to(root)).lower()
        key = re.sub(r"\.nii(?:\.gz)?$", "", key)
        unique.setdefault(key, path)
    return list(unique.values())


def _local_patient_paths(patient_number: int) -> tuple[Path, list[Path]]:
    patient = f"Patient_{patient_number:02d}"
    candidates = [
        path
        for path in _unique_nifti_files(DEFAULT_MTR_DIR)
        if patient.lower() in path.name.lower()
    ]
    if not candidates:
        raise ValueError(f"No default MTR volume found for {patient}")
    candidates.sort(key=lambda path: ("masked" not in path.name.lower(), path.name.lower()))
    segmentation_dir = DEFAULT_SEGMENTATION_DIR / f"Segmentation {patient_number}"
    segmentations = _unique_nifti_files(segmentation_dir) if segmentation_dir.is_dir() else []
    if not segmentations:
        raise ValueError(f"No default segmentations found for {patient}")
    return candidates[0], segmentations


def _result_urls(summary: dict[str, object], job_id: str) -> dict[str, object]:
    summary["angle_map_url"] = f"/results/{job_id}/combined_tendon_angle_degrees.nii.gz"
    summary["magic_map_url"] = f"/results/{job_id}/combined_magic_angle_parameter.nii.gz"
    summary["magic_deviation_map_url"] = (
        f"/results/{job_id}/combined_magic_angle_deviation_degrees.nii.gz"
    )
    summary["label_map_url"] = f"/results/{job_id}/combined_tendon_labels.nii.gz"
    summary["centerline_url"] = f"/results/{job_id}/combined_tendon_centerlines.nii.gz"
    summary["summary_csv_url"] = f"/results/{job_id}/tendon_angle_summary.csv"
    return summary


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/local-data")
def local_data():
    patients = []
    for patient_number in range(1, 100):
        try:
            mtr_path, segmentation_paths = _local_patient_paths(patient_number)
        except ValueError:
            continue
        patients.append(
            {
                "number": patient_number,
                "patient": f"Patient_{patient_number:02d}",
                "mtr_name": mtr_path.name,
                "mtr_masked": "masked" in mtr_path.name.lower(),
                "segmentation_count": len(segmentation_paths),
                "segmentation_names": [path.name for path in segmentation_paths],
            }
        )
    return jsonify(
        patients=patients,
        segmentation_root=str(DEFAULT_SEGMENTATION_DIR),
        mtr_root=str(DEFAULT_MTR_DIR),
    )


@app.get("/api/orientation-local/<int:patient_number>")
def orientation_local(patient_number: int):
    try:
        mtr_path, _ = _local_patient_paths(patient_number)
        result = dicom_orientation_for_filename(mtr_path.name)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    return jsonify(filename=mtr_path.name, **result)


@app.get("/api/local-volume/<int:patient_number>/<kind>/<int:index>")
def local_volume(patient_number: int, kind: str, index: int):
    try:
        mtr_path, segmentation_paths = _local_patient_paths(patient_number)
        if kind == "mtr" and index == 0:
            path = mtr_path
        elif kind == "segmentation" and 0 <= index < len(segmentation_paths):
            path = segmentation_paths[index]
        else:
            raise ValueError("Unknown local volume")
    except ValueError as error:
        return jsonify(error=str(error)), 404
    return send_from_directory(path.parent, path.name, as_attachment=False)


@app.post("/api/analyze-local/<int:patient_number>")
def analyze_local(patient_number: int):
    try:
        reference_path, input_paths = _local_patient_paths(patient_number)
        b0 = np.array([float(request.form.get(axis, default)) for axis, default in zip(("b0x", "b0y", "b0z"), (0, 0, 1))])
        if not np.isfinite(b0).all() or np.linalg.norm(b0) == 0:
            raise ValueError("B0 must be a finite, non-zero vector")
        job_id = uuid.uuid4().hex
        summary = analyze_segmentations(
            input_paths,
            RESULTS_DIR / job_id,
            b0,
            b0_basis="image",
            reference_path=reference_path,
            segmentation_names=[path.name for path in input_paths],
        )
    except (OSError, ValueError) as error:
        return jsonify(error=str(error)), 400
    return jsonify(_result_urls(summary, job_id))


@app.post("/api/orientation")
def orientation():
    upload = request.files.get("volume")
    if upload is None or not upload.filename:
        return jsonify(error="Choose a NIfTI volume first."), 400
    filename = secure_filename(upload.filename)
    if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
        return jsonify(error="Volume must be a .nii or .nii.gz file."), 400
    try:
        result = dicom_orientation_for_filename(filename)
    except ValueError as error:
        return jsonify(error=str(error)), 400
    return jsonify(filename=filename, **result)


@app.post("/api/analyze")
def analyze():
    uploads = request.files.getlist("segmentations")
    reference_upload = request.files.get("reference")
    if not uploads or any(not upload.filename for upload in uploads):
        return jsonify(error="Choose a folder containing NIfTI segmentations."), 400
    if reference_upload is None or not reference_upload.filename:
        return jsonify(error="Choose the subject's reference MTR volume."), 400
    display_names = [
        Path(upload.filename.replace("\\", "/")).name for upload in uploads
    ]
    filenames = [secure_filename(upload.filename) for upload in uploads]
    if any(not name.lower().endswith((".nii", ".nii.gz")) for name in filenames):
        return jsonify(error="Every segmentation must be a .nii or .nii.gz file."), 400
    try:
        b0 = np.array([float(request.form.get(axis, default)) for axis, default in zip(("b0x", "b0y", "b0z"), (0, 0, 1))])
        if not np.isfinite(b0).all() or np.linalg.norm(b0) == 0:
            raise ValueError("B0 must be a finite, non-zero vector")
        job_id = uuid.uuid4().hex
        job_dir = RESULTS_DIR / job_id
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            reference_filename = secure_filename(reference_upload.filename)
            reference_path = temporary_root / reference_filename
            reference_upload.save(reference_path)
            input_paths = []
            for index, (upload, filename) in enumerate(zip(uploads, filenames), start=1):
                input_dir = temporary_root / f"segmentation_{index:02d}"
                input_dir.mkdir()
                input_path = input_dir / filename
                upload.save(input_path)
                input_paths.append(input_path)
            summary = analyze_segmentations(
                input_paths,
                job_dir,
                b0,
                b0_basis="image",
                reference_path=reference_path,
                segmentation_names=display_names,
            )
    except (OSError, ValueError) as error:
        return jsonify(error=str(error)), 400

    return jsonify(_result_urls(summary, job_id))


@app.get("/results/<job_id>/<path:filename>")
def result(job_id: str, filename: str):
    return send_from_directory(RESULTS_DIR / secure_filename(job_id), filename, as_attachment=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Niivue tendon orientation viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not args.no_browser:
        Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
