from pathlib import Path
import unittest

from viewer.time_series import derive_time_series_path, PROCESSED_ROOT


class DeriveTimeSeriesPathTests(unittest.TestCase):
    def test_basic_mapping(self) -> None:
        video = Path(r"D:\dataset\drowsy_driving_raja\S1\MD.mff.S01_20170519_043933.mov")
        expected = PROCESSED_ROOT / "S1" / "S01_20170519_043933" / "ear_eog.fif"
        self.assertEqual(derive_time_series_path(video), expected)

    def test_suffix_stripping(self) -> None:
        video = Path(r"D:\dataset\drowsy_driving_raja\S2\MD.mff.S02_20170519_043933_2.mov")
        expected = PROCESSED_ROOT / "S2" / "S02_20170519_043933" / "ear_eog.fif"
        self.assertEqual(derive_time_series_path(video), expected)

    def test_nested_subject_folder(self) -> None:
        video = Path(r"D:\dataset\drowsy_driving_raja\S3\sessions\MD.mff.S03_20170519_043933.mov")
        expected = PROCESSED_ROOT / "S3" / "S03_20170519_043933" / "ear_eog.fif"
        self.assertEqual(derive_time_series_path(video), expected)

    def test_raises_when_no_subject_found(self) -> None:
        video = Path(r"D:\dataset\drowsy_driving_raja\unknown\MD.mff.S99_20170519_043933.mov")
        with self.assertRaises(ValueError):
            derive_time_series_path(video)


if __name__ == "__main__":
    unittest.main()
