import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from dicom_orientation import acute_angle_degrees
from tendon_geometry import (
    _grow_mask_by_mtr,
    _primary_centerline_path,
    analyze_segmentation,
    analyze_segmentations,
    world_vector_in_image_axes,
)


class TendonGeometryTests(unittest.TestCase):
    def test_primary_centerline_uses_trunk_of_branched_skeleton(self):
        trunk = [(x, 2, 2) for x in range(9)]
        branch = [(4, y, 2) for y in range(3, 6)]
        indices = np.asarray(trunk + branch)

        path, endpoint_count, branch_count = _primary_centerline_path(
            indices, indices.astype(float)
        )
        selected = indices[path]

        self.assertGreaterEqual(endpoint_count, 3)
        self.assertGreater(branch_count, 0)
        self.assertEqual(tuple(selected[0]), (0, 2, 2))
        self.assertEqual(tuple(selected[-1]), (8, 2, 2))
        self.assertTrue(np.all(selected[:, 1] == 2))

    def test_acute_angle_is_direction_independent(self):
        self.assertEqual(acute_angle_degrees(np.array([0, 0, 1]), np.array([0, 0, -1])), 0.0)
        self.assertEqual(acute_angle_degrees(np.array([0, 0, 1]), np.array([1, 0, 0])), 90.0)

    def test_world_vector_is_expressed_in_oblique_image_axes(self):
        angle = np.radians(10.0)
        affine = np.eye(4)
        affine[:3, :3] = np.array(
            [[1, 0, 0], [0, np.cos(angle), np.sin(angle)], [0, -np.sin(angle), np.cos(angle)]]
        )

        result = world_vector_in_image_axes(affine, np.array([0, 0, 1]))

        np.testing.assert_allclose(result, [0, -np.sin(angle), np.cos(angle)], atol=1e-7)

    def test_straight_centerline_angle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((15, 15, 30), dtype=np.uint8)
            mask[6:9, 6:9, 3:27] = 1
            segmentation = root / "mask.nii.gz"
            nib.save(nib.Nifti1Image(mask, np.eye(4)), segmentation)

            summary = analyze_segmentation(segmentation, root / "output")

            self.assertLess(summary["median_angle_deg"], 1.0)
            self.assertGreater(len(summary["profile_position_mm"]), 1)
            self.assertEqual(
                len(summary["profile_position_mm"]),
                len(summary["profile_angle_deg"]),
            )
            self.assertTrue(all(value < 1.0 for value in summary["profile_angle_deg"]))
            self.assertTrue((root / "output" / "tendon_angle_degrees.nii.gz").exists())
            self.assertTrue((root / "output" / "tendon_centerline.nii.gz").exists())

    def test_meaningful_fragments_are_preserved_and_specks_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((15, 15, 30), dtype=np.uint8)
            mask[5:8, 5:8, 2:12] = 1
            mask[5:8, 5:8, 14:24] = 1
            mask[0, 0, 0] = 1
            segmentation = root / "fragmented_mask.nii.gz"
            nib.save(nib.Nifti1Image(mask, np.eye(4)), segmentation)

            summary = analyze_segmentation(segmentation, root / "output")
            angle_map = nib.load(root / "output" / "tendon_angle_degrees.nii.gz").get_fdata()

            self.assertEqual(summary["input_foreground_voxels"], 181)
            self.assertEqual(summary["foreground_voxels"], 180)
            self.assertEqual(summary["excluded_small_component_voxels"], 1)
            self.assertTrue(np.isfinite(angle_map[5:8, 5:8, 2:12]).all())
            self.assertTrue(np.isfinite(angle_map[5:8, 5:8, 14:24]).all())
            self.assertTrue(np.isnan(angle_map[0, 0, 0]))

    def test_region_growing_adds_only_nearby_high_mtr_voxels(self):
        mask = np.zeros((15, 15, 30), dtype=bool)
        mask[6:9, 6:9, 3:27] = True
        mtr = np.zeros(mask.shape, dtype=np.float32)
        mtr[mask] = 0.5
        mtr[5, 6:9, 3:27] = 0.6
        mtr[9, 6:9, 3:27] = 0.2

        grown, threshold = _grow_mask_by_mtr(
            mask, mtr, (1.0, 1.0, 1.0), None, 1.0, 25.0
        )

        self.assertEqual(int(grown.sum() - mask.sum()), 72)
        self.assertAlmostEqual(threshold, 0.5)
        self.assertTrue(grown[5, 6:9, 3:27].all())
        self.assertFalse(grown[9, 6:9, 3:27].any())

    def test_combines_tendons_into_angle_magic_and_centerline_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = np.zeros((20, 20, 30), dtype=np.uint8)
            second = np.zeros_like(first)
            first[3:6, 3:6, 3:27] = 1
            second[13:16, 13:16, 3:27] = 1
            paths = [root / "first.nii.gz", root / "second.nii.gz"]
            for path, mask in zip(paths, (first, second)):
                nib.save(nib.Nifti1Image(mask, np.eye(4)), path)

            summary = analyze_segmentations(
                paths,
                root / "output",
                segmentation_names=["First tendon.nii.gz", "Second tendon.nii.gz"],
            )
            angles = nib.load(
                root / "output" / "combined_tendon_angle_degrees.nii.gz"
            ).get_fdata()
            magic = nib.load(
                root / "output" / "combined_magic_angle_parameter.nii.gz"
            ).get_fdata()
            deviation = nib.load(
                root / "output" / "combined_magic_angle_deviation_degrees.nii.gz"
            ).get_fdata()
            centerlines = nib.load(
                root / "output" / "combined_tendon_centerlines.nii.gz"
            ).get_fdata()

            self.assertEqual(summary["tendon_count"], 2)
            self.assertTrue(np.isfinite(angles[first.astype(bool)]).all())
            self.assertTrue(np.isfinite(angles[second.astype(bool)]).all())
            np.testing.assert_allclose(magic[np.isfinite(magic)], 2.0, atol=1e-6)
            np.testing.assert_allclose(
                deviation[np.isfinite(deviation)], 54.7356, atol=1e-4
            )
            self.assertEqual(set(np.unique(centerlines)), {0.0, 1.0, 2.0})
            with (root / "output" / "tendon_angle_summary.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["segmentation"], "First tendon.nii.gz")

    def test_splits_distinct_labels_within_one_segmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.zeros((20, 20, 30), dtype=np.uint8)
            labels[3:6, 3:6, 3:27] = 1
            labels[13:16, 13:16, 3:27] = 4
            path = root / "multi_label.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), path)

            summary = analyze_segmentations([path], root / "output")
            combined_labels = nib.load(
                root / "output" / "combined_tendon_labels.nii.gz"
            ).get_fdata()

            self.assertEqual(summary["tendon_count"], 2)
            self.assertEqual([item["source_label"] for item in summary["tendons"]], [1, 4])
            self.assertEqual(set(np.unique(combined_labels)), {0.0, 1.0, 2.0})


if __name__ == "__main__":
    unittest.main()
