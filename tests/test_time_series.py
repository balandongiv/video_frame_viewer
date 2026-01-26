import unittest
from pathlib import Path

from config import AppConfig
from paths import derive_annotation_path, derive_time_series_path


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


if __name__ == "__main__":
    unittest.main()
