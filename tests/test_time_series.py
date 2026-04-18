import unittest
from pathlib import Path

import numpy as np

from config import AppConfig
from paths import derive_annotation_path, derive_time_series_path
from time_series import TimeSeriesViewer


class DeriveTimeSeriesPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(dataset_root=Path(r"D:\dataset\drowsy_driving_raja"))

    def test_basic_mapping(self) -> None:
        video = Path(r"D:\dataset\drowsy_driving_raja\S1\MD.mff.S01_20170519_043933.mov")
        expected = (
            self.config.time_series_root / "S1" / "S01_20170519_043933" / "ear_eog.fif"
        )
        self.assertEqual(
            derive_time_series_path(video, processed_root=self.config.time_series_root),
            expected,
        )

    def test_keeps_segment_suffix_in_folder_name(self) -> None:
        video = Path(r"D:\dataset\drowsy_driving_raja\S2\MD.mff.S02_20170519_043933_2.mov")
        expected = (
            self.config.time_series_root / "S2" / "S02_20170519_043933_2" / "ear_eog.fif"
        )
        self.assertEqual(
            derive_time_series_path(video, processed_root=self.config.time_series_root),
            expected,
        )

    def test_keeps_higher_segment_suffix_in_folder_name(self) -> None:
        video = Path(r"D:\dataset\drowsy_driving_raja\S11\MD.mff\S24_20181227_034657_3.mov")
        expected = (
            self.config.time_series_root / "S11" / "S24_20181227_034657_3" / "ear_eog.fif"
        )
        self.assertEqual(
            derive_time_series_path(video, processed_root=self.config.time_series_root),
            expected,
        )

    def test_nested_subject_folder(self) -> None:
        video = Path(r"D:\dataset\drowsy_driving_raja\S3\sessions\MD.mff.S03_20170519_043933.mov")
        expected = (
            self.config.time_series_root / "S3" / "S03_20170519_043933" / "ear_eog.fif"
        )
        self.assertEqual(
            derive_time_series_path(video, processed_root=self.config.time_series_root),
            expected,
        )

    def test_raises_when_no_subject_found(self) -> None:
        video = Path(r"D:\dataset\drowsy_driving_raja\unknown\MD.mff.S99_20170519_043933.mov")
        with self.assertRaises(ValueError):
            derive_time_series_path(video, processed_root=self.config.time_series_root)

    def test_annotation_path_uses_custom_csv_dir(self) -> None:
        config = AppConfig(dataset_root=self.config.dataset_root, csv_dir="annotations")
        video = Path(r"D:\dataset\drowsy_driving_raja\S1\MD.mff.S01_20170519_043933.mov")
        annotation_path = derive_annotation_path(
            video,
            processed_root=config.time_series_root,
            csv_root=config.annotation_root,
        )
        expected = config.annotation_root / "S1" / "S01_20170519_043933" / "ear_eog.csv"
        self.assertEqual(annotation_path, expected)

    def test_annotation_path_keeps_segment_suffix(self) -> None:
        video = Path(r"D:\dataset\drowsy_driving_raja\S1\MD.mff\S01_20170519_043933_2.mov")
        annotation_path = derive_annotation_path(
            video,
            processed_root=self.config.time_series_root,
            csv_root=self.config.annotation_root,
        )
        expected = (
            self.config.annotation_root / "S1" / "S01_20170519_043933_2" / "ear_eog.csv"
        )
        self.assertEqual(annotation_path, expected)


class AutoRepairAnnotationTests(unittest.TestCase):
    def test_uses_peak_and_nearest_zero_crossings(self) -> None:
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        data = np.array([-1.0, 1.0, 5.0, 1.0, -1.0])

        repaired = TimeSeriesViewer._repair_bounds_from_samples(times, data)

        self.assertEqual(repaired, (0.05, 0.35, 0.2))

    def test_repairs_overshot_annotation_to_zero_crossings_around_peak(self) -> None:
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        data = np.array([-1.0, 1.0, -1.0, 1.0, 6.0, 1.0, -1.0, 1.0])

        repaired = TimeSeriesViewer._repair_bounds_from_samples(times, data)

        self.assertEqual(repaired, (0.25, 0.55, 0.4))

    def test_can_find_crossings_outside_current_annotation_window(self) -> None:
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        data = np.array([-1.0, -0.5, 2.0, 5.0, 2.0, -1.0])

        repaired = TimeSeriesViewer._repair_bounds_from_peak(times, data, peak_index=3)

        self.assertEqual(repaired, (0.12, 0.4666666666666667, 0.3))

    def test_does_not_borrow_neighbor_zero_crossing(self) -> None:
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        data = np.array([-1.0, 1.0, 4.0, 1.0, 1.0, 2.0, 1.0, -1.0])

        repaired = TimeSeriesViewer._repair_bounds_from_peak(
            times,
            data,
            peak_index=2,
            search_start_index=0,
            search_end_index=4,
        )

        self.assertIsNone(repaired)

    def test_rejects_when_peak_has_no_positive_zero_crossing_lobe(self) -> None:
        times = np.array([0.0, 0.1, 0.2])
        data = np.array([-1.0, -0.5, -1.0])

        self.assertIsNone(TimeSeriesViewer._repair_bounds_from_samples(times, data))


if __name__ == "__main__":
    unittest.main()
