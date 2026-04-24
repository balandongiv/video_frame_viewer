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

        self.assertIsNotNone(repaired)
        assert repaired is not None
        onset, end, event_time = repaired
        self.assertAlmostEqual(onset, 0.12)
        self.assertAlmostEqual(end, 0.4666666666666667)
        self.assertAlmostEqual(event_time, 0.3)

    def test_uses_configured_repair_threshold_for_crossings(self) -> None:
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        data = np.array([0.0, 2.0, 6.0, 2.0, 0.0])

        repaired = TimeSeriesViewer._repair_bounds_from_samples(
            times, data, threshold=1.0
        )

        self.assertEqual(repaired, (0.05, 0.35, 0.2))

    def test_negative_threshold_uses_trough_and_matching_crossings(self) -> None:
        times = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        data = np.array([0.0, -5.0, -15.0, -5.0, 0.0])

        repaired = TimeSeriesViewer._repair_bounds_from_samples(
            times, data, threshold=-9.6
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        onset, end, event_time = repaired
        self.assertAlmostEqual(onset, 0.146)
        self.assertAlmostEqual(end, 0.254)
        self.assertAlmostEqual(event_time, 0.2)

    def test_eog_repair_uses_local_threshold_segment_not_larger_later_peak(self) -> None:
        times = np.array(
            [15.60, 15.62, 15.63, 15.64, 15.70, 15.73, 15.74, 15.78, 15.79, 15.80]
        )
        data = np.array(
            [11.8, -15.1, -27.3, -30.8, -38.8, -18.0, -15.0, -60.0, -95.0, -15.0]
        )

        repaired = TimeSeriesViewer._repair_eog_bounds_from_threshold_segment(
            times,
            data,
            annotation_onset=15.466667,
            annotation_duration=0.166667,
            threshold=-17.3,
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        onset, end, event_time = repaired
        self.assertAlmostEqual(onset, 15.63)
        self.assertAlmostEqual(end, 15.73)
        self.assertAlmostEqual(event_time, 15.70)
        self.assertLessEqual(onset, event_time)
        self.assertLessEqual(event_time, end)

    def test_eog_repair_selects_nearest_threshold_segment_after_early_label(self) -> None:
        times = np.array([18.70, 18.72, 18.73, 18.76, 18.83, 18.84, 18.88, 18.89])
        data = np.array([3.8, 7.6, -18.0, -24.0, -28.1, -16.0, -80.0, -10.0])

        repaired = TimeSeriesViewer._repair_eog_bounds_from_threshold_segment(
            times,
            data,
            annotation_onset=18.5,
            annotation_duration=0.233333,
            threshold=-17.2,
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        onset, end, event_time = repaired
        self.assertAlmostEqual(onset, 18.73)
        self.assertAlmostEqual(end, 18.83)
        self.assertAlmostEqual(event_time, 18.83)
        self.assertLessEqual(onset, event_time)
        self.assertLessEqual(event_time, end)

    def test_eog_repair_does_not_bridge_neighboring_threshold_segments(self) -> None:
        times = np.array(
            [15.60, 15.63, 15.70, 15.73, 15.76, 15.83, 15.90, 15.93, 15.96]
        )
        data = np.array([5.0, -20.0, -31.0, -18.0, 4.0, -21.0, -35.0, -19.0, 3.0])

        repaired = TimeSeriesViewer._repair_eog_bounds_from_threshold_segment(
            times,
            data,
            annotation_onset=15.466667,
            annotation_duration=0.166667,
            threshold=-17.3,
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        onset, end, event_time = repaired
        self.assertAlmostEqual(onset, 15.63)
        self.assertAlmostEqual(end, 15.73)
        self.assertAlmostEqual(event_time, 15.70)
        self.assertLessEqual(onset, event_time)
        self.assertLessEqual(event_time, end)

    def test_eog_repair_prefers_segment_containing_local_peak_anchor(self) -> None:
        times = np.array([1.00, 1.03, 1.06, 1.10, 1.13, 1.16, 1.20])
        data = np.array([2.0, -18.0, -19.0, 3.0, -18.5, -35.0, -19.0])

        repaired = TimeSeriesViewer._repair_eog_bounds_from_threshold_segment(
            times,
            data,
            annotation_onset=1.0,
            annotation_duration=0.14,
            threshold=-17.3,
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        onset, end, event_time = repaired
        self.assertAlmostEqual(onset, 1.13)
        self.assertAlmostEqual(end, 1.20)
        self.assertAlmostEqual(event_time, 1.16)
        self.assertLessEqual(onset, event_time)
        self.assertLessEqual(event_time, end)

    def test_eog_repair_keeps_one_event_across_roundoff_threshold_jitter(self) -> None:
        times = np.array([15.63, 15.64, 15.65, 15.70, 15.73, 15.76])
        data = np.array([-17.306, -17.299, -17.316, -17.338, -17.337, -16.9])

        repaired = TimeSeriesViewer._repair_eog_bounds_from_threshold_segment(
            times,
            data,
            annotation_onset=15.466667,
            annotation_duration=0.166667,
            threshold=-17.3,
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        onset, end, event_time = repaired
        self.assertAlmostEqual(onset, 15.63)
        self.assertAlmostEqual(end, 15.73)
        self.assertAlmostEqual(event_time, 15.70)
        self.assertLessEqual(onset, event_time)
        self.assertLessEqual(event_time, end)

    def test_eog_repair_matches_s12_peak_anchored_threshold_window(self) -> None:
        times = np.array(
            [
                15.59,
                15.60,
                15.61,
                15.62,
                15.63,
                15.64,
                15.65,
                15.70,
                15.71,
                15.72,
                15.73,
                15.74,
                15.75,
            ]
        )
        data = np.array(
            [
                -17.209653,
                -17.226579,
                -17.271510,
                -17.265433,
                -17.306463,
                -17.298897,
                -17.316652,
                -17.312940,
                -17.337680,
                -17.316146,
                -17.337123,
                -17.317483,
                -17.336997,
            ]
        )

        repaired = TimeSeriesViewer._repair_eog_bounds_from_threshold_segment(
            times,
            data,
            annotation_onset=15.466667,
            annotation_duration=0.166667,
            threshold=-17.2,
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        onset, end, event_time = repaired
        self.assertAlmostEqual(onset, 15.63)
        self.assertAlmostEqual(end, 15.73)
        self.assertAlmostEqual(event_time, 15.71)
        self.assertLessEqual(onset, event_time)
        self.assertLessEqual(event_time, end)

    def test_eog_repair_finds_peak_before_annotation_onset(self) -> None:
        # Runtime case: annotation onset is at the event's right threshold crossing;
        # the actual peak is slightly before the onset. The repair must still find
        # the correct segment and satisfy new_onset <= peak_time <= new_end.
        times = np.array(
            [15.59, 15.60, 15.63, 15.65, 15.68, 15.71, 15.73, 15.75, 15.77, 15.80]
        )
        data = np.array(
            [3.0, 2.0, -18.5, -22.0, -28.0, -35.0, -18.2, 4.0, 5.0, 3.0]
        )

        repaired = TimeSeriesViewer._repair_eog_bounds_from_threshold_segment(
            times,
            data,
            annotation_onset=15.73,
            annotation_duration=0.10,
            threshold=-17.3,
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        onset, end, event_time = repaired
        self.assertAlmostEqual(onset, 15.63)
        self.assertAlmostEqual(end, 15.73)
        self.assertAlmostEqual(event_time, 15.71)
        self.assertLessEqual(onset, event_time)
        self.assertLessEqual(event_time, end)

    def test_eog_repair_data_uses_left_channel_only(self) -> None:
        class FakeRaw:
            ch_names = ["EOG-EEG-eog_vert_left", "EOG-EEG-eog_vert_right"]
            info = {"chs": [{}, {}]}
            _orig_units: dict[str, str] = {}

            def __init__(self) -> None:
                self._data = [
                    np.array([1.0, 2.0, 3.0]),
                    np.array([10.0, 20.0, 30.0]),
                ]

            def get_data(self, picks, verbose):
                return np.array([self._data[picks[0]]])

            def get_channel_types(self, picks):
                return ["misc" for _ in picks]

        viewer = TimeSeriesViewer.__new__(TimeSeriesViewer)
        viewer.raw = FakeRaw()

        repaired_data = viewer._eog_repair_data()

        np.testing.assert_array_equal(repaired_data, np.array([1.0, 2.0, 3.0]))

    def test_plot_channel_indices_only_include_required_channels(self) -> None:
        class FakeRaw:
            ch_names = [
                "EEG-F3",
                "EAR-avg_ear",
                "EOG-EEG-eog_vert_right",
                "EOG-EEG-eog_vert_left",
                "EEG-E8",
                "EEG-P7",
            ]

        viewer = TimeSeriesViewer.__new__(TimeSeriesViewer)
        viewer.raw = FakeRaw()

        self.assertEqual(viewer._channel_indices(), [1, 3, 4])

    def test_plot_channel_order_does_not_duplicate_highlighted_ear(self) -> None:
        viewer = TimeSeriesViewer.__new__(TimeSeriesViewer)
        viewer._primary_channel = "EAR-avg_ear"
        data = np.arange(9).reshape(3, 3)
        picks = [1, 3, 4]
        channel_names = ["EAR-avg_ear", "EOG-EEG-eog_vert_left", "EEG-E8"]
        channel_types = ["misc", "eog", "eeg"]

        ordered = viewer._order_channels_for_display(
            data, picks, channel_names, channel_types
        )

        self.assertEqual(ordered[1], picks)
        self.assertEqual(ordered[2], channel_names)

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
