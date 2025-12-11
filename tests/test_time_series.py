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


if __name__ == "__main__":
    unittest.main()
