# MTR tendon orientation

Minimal local tools for auditing DICOM orientation and measuring tendon centerline angles to $B_0$.

## Run

```powershell
.\.venv\Scripts\python.exe dicom_orientation.py "Y:\_collab_raw\Raw Data UTE MT Foot\dicom_pacs"
.\.venv\Scripts\python.exe app.py
```

The viewer opens at `http://127.0.0.1:5000`. Load a map named with its patient and map type, such as `Patient_27_mtr.nii.gz`, then choose that subject's segmentation folder. All unique `.nii`/`.nii.gz` masks in the folder are loaded; duplicate compressed and uncompressed copies are processed once. The viewer overlays the source masks, combined voxelwise angle map, and labeled centerlines for visual quality control.

## GitHub Pages

The data-free static analyzer is in `docs/index.html`. It runs the shared `tendon_geometry.py` pipeline in a Pyodide Web Worker and accepts a local MTR image, one or more tendon segmentation NIfTI files, and the B0 vector in image axes. Selected files are not uploaded. The generated angle, magic-angle, deviation, label, centerline, and CSV outputs can be viewed and downloaded directly in the browser. Precomputed outputs can still be opened independently.

The Pages workflow deploys `docs/` after pushes to `main`. In the GitHub repository settings, choose **Settings > Pages > Source > GitHub Actions**. Imaging data, generated results, and the generated DICOM orientation CSV are excluded by `.gitignore`.

Each positive integer label in a segmentation file is treated as a separate tendon segment. The viewer colors these segments distinctly after analysis, even when several labels are stored in one NIfTI file. The downloads contain one combined angle map in degrees, one combined magic-angle parameter map, one absolute magic-angle deviation map, one labeled tendon map, one labeled centerline map, and a per-tendon summary CSV for the subject. Background voxels are `NaN` in the floating-point maps. The magic-angle parameter is $3\cos^2(\theta)-1$, with range $[-1, 2]$. Segmentations must share the reference MTR shape and affine and must not overlap.

The angle map stores the acute angle $\theta$ between the local tendon centerline tangent and $B_0$, from 0 to 90 degrees. The parameter map stores $3\cos^2(\theta)-1$. The deviation map stores $|\theta-54.7356^\circ|$ and is displayed red near the magic angle and green far from it. Centerlines can be toggled over any map, and the viewer provides both 3D and combined orthogonal-plus-3D layouts.

## Method

The exporter reads `ImageOrientationPatient` independently from MTon (the source array/header used by `mtr_all.m`) and T2star (the source array/header used by `t2s_all.m`). $B_0$ is scanner Z. The map scripts do not permute their source arrays, and the dcm2niix conversion used here stores these sagittal arrays as `[DICOM slice, -DICOM row, -DICOM column]`. Therefore scanner Z is projected into that map basis directly from DICOM direction cosines. No NIfTI affine or orientation field participates. Sign is immaterial because the magic-angle relation depends on `cos²(theta)`.

For each binary tendon mask, connected components smaller than 20 voxels are treated as specks. The cleaned mask grows by at most 1 mm into connected voxels above that tendon's 75th-percentile MTR, while original neighboring tendon labels remain reserved. The resulting mask is skeletonized in 3D and converted to a weighted 26-neighbor graph. The longest geodesic endpoint path is retained as the primary centerline, excluding side branches from longitudinal profiles. Local tangent vectors are estimated by PCA over primary-path points within 5 mm, then assigned to mask voxels by nearest-centerline lookup. The output angle is constrained to 0-90 degrees. Profile distance is cumulative distance from a consistently oriented path endpoint, not an anatomical insertion landmark. Endpoint and branch counts are reported for topology QC.