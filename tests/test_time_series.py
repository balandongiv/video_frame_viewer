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

    def test_eog_peak_trough_matches_user_example(self) -> None:
        # Peak=10 at index 5; left trough=-1 at index 1 (4 samples left, accepted).
        # Right trough=4 at index 8 is only 3 samples away so it is skipped;
        # the walk continues to the data edge at index 9.
        times = np.arange(10, dtype=float)
        data = np.array([0.0, -1.0, 0.0, 2.0, 3.0, 10.0, 9.0, 5.0, 4.0, 9.0])

        repaired = TimeSeriesViewer._repair_eog_bounds_from_peak_trough(
            times, data, annotation_onset=0.0, annotation_duration=10.0
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        left_time, right_time, peak_time = repaired
        self.assertAlmostEqual(peak_time, 5.0)
        self.assertAlmostEqual(left_time, 1.0)
        self.assertAlmostEqual(right_time, 9.0)

    def test_eog_find_left_trough_finds_local_minimum(self) -> None:
        # Trough at index 1 is exactly 4 samples from peak at index 5 — accepted.
        data = np.array([0.0, -1.0, 0.0, 2.0, 3.0, 10.0, 9.0, 5.0, 4.0, 9.0])
        self.assertEqual(TimeSeriesViewer._eog_find_left_trough_index(data, 5), 1)

    def test_eog_find_right_trough_finds_local_minimum(self) -> None:
        # Trough at index 8 is exactly 4 samples from peak at index 4 — accepted.
        data = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 8.0, 6.0, 4.0, 2.0, 3.0])
        self.assertEqual(TimeSeriesViewer._eog_find_right_trough_index(data, 4), 8)

    def test_eog_trough_skips_turning_point_closer_than_min_separation(self) -> None:
        # Right trough at index 8 is 3 samples from peak at index 5 — too close,
        # so the walk continues to the data edge at index 9.
        data = np.array([0.0, -1.0, 0.0, 2.0, 3.0, 10.0, 9.0, 5.0, 4.0, 9.0])
        self.assertEqual(TimeSeriesViewer._eog_find_right_trough_index(data, 5), 9)

    def test_eog_find_left_trough_returns_zero_when_no_turn(self) -> None:
        # Monotonically increasing going leftward: no local minimum before the peak.
        data = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        self.assertEqual(TimeSeriesViewer._eog_find_left_trough_index(data, 4), 0)

    def test_eog_find_right_trough_returns_last_when_no_turn(self) -> None:
        data = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
        self.assertEqual(TimeSeriesViewer._eog_find_right_trough_index(data, 0), 4)

    def test_eog_peak_trough_extends_beyond_annotation_for_left_boundary(self) -> None:
        # Annotation covers only the peak and the rising right side;
        # the left trough must be found outside the annotation window.
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        data = np.array([-1.0, 0.0, 10.0, 8.0, 5.0, 3.0])
        # Annotation window starts at t=2 (peak), so left trough at index 0 is outside
        repaired = TimeSeriesViewer._repair_eog_bounds_from_peak_trough(
            times, data, annotation_onset=2.0, annotation_duration=4.0
        )
        self.assertIsNotNone(repaired)
        assert repaired is not None
        left_time, right_time, peak_time = repaired
        self.assertAlmostEqual(peak_time, 2.0)
        self.assertAlmostEqual(left_time, 0.0)
        self.assertAlmostEqual(right_time, 5.0)

    def test_eog_peak_trough_returns_none_when_annotation_outside_data(self) -> None:
        times = np.arange(10, dtype=float)
        data = np.ones(10)
        result = TimeSeriesViewer._repair_eog_bounds_from_peak_trough(
            times, data, annotation_onset=20.0, annotation_duration=5.0
        )
        self.assertIsNone(result)

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

    # --- EAR repair tests ---

    def test_ear_find_left_peak_example_from_spec(self) -> None:
        # Signal from spec: 290, 390, 300, 250, 100, 150, 200, 500, 400
        # min at index 4; 1st left peak at index 1 (value 390) because data[0]=290 < data[1]=390
        data = np.array([290.0, 390.0, 300.0, 250.0, 100.0, 150.0, 200.0, 500.0, 400.0])
        self.assertEqual(TimeSeriesViewer._ear_find_left_peak_index(data, 4, nth_peak=1), 1)

    def test_ear_find_right_peak_example_from_spec(self) -> None:
        # Signal from spec: 290, 390, 300, 250, 100, 150, 200, 500, 400
        # min at index 4; 1st right peak at index 7 (value 500) because data[8]=400 < data[7]=500
        data = np.array([290.0, 390.0, 300.0, 250.0, 100.0, 150.0, 200.0, 500.0, 400.0])
        self.assertEqual(TimeSeriesViewer._ear_find_right_peak_index(data, 4, nth_peak=1), 7)

    def test_ear_find_left_peak_returns_zero_when_no_turn(self) -> None:
        # Monotonically decreasing going leftward: no local maximum before the minimum.
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        self.assertEqual(TimeSeriesViewer._ear_find_left_peak_index(data, 4, nth_peak=1), 0)

    def test_ear_find_right_peak_returns_last_when_no_turn(self) -> None:
        # Monotonically increasing going rightward: no local maximum after the minimum.
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(TimeSeriesViewer._ear_find_right_peak_index(data, 0, nth_peak=1), 4)

    def test_repair_ear_bounds_from_trough_peak_spec_example(self) -> None:
        # Full spec example: annotation covers entire signal.
        # min=100 at index 4; left=390 at index 1; right=500 at index 7
        times = np.arange(9, dtype=float)
        data = np.array([290.0, 390.0, 300.0, 250.0, 100.0, 150.0, 200.0, 500.0, 400.0])
        repaired = TimeSeriesViewer._repair_ear_bounds_from_trough_peak(
            times, data, annotation_onset=0.0, annotation_duration=9.0, nth_peak=1
        )
        self.assertIsNotNone(repaired)
        assert repaired is not None
        left_time, right_time, min_time = repaired
        self.assertAlmostEqual(min_time, 4.0)
        self.assertAlmostEqual(left_time, 1.0)
        self.assertAlmostEqual(right_time, 7.0)

    def test_repair_ear_bounds_extends_beyond_annotation(self) -> None:
        # Annotation covers only the valley; peaks are outside the annotation window.
        times = np.arange(7, dtype=float)
        data = np.array([500.0, 300.0, 100.0, 50.0, 100.0, 300.0, 500.0])
        repaired = TimeSeriesViewer._repair_ear_bounds_from_trough_peak(
            times, data, annotation_onset=2.0, annotation_duration=2.0, nth_peak=1
        )
        self.assertIsNotNone(repaired)
        assert repaired is not None
        left_time, right_time, min_time = repaired
        self.assertAlmostEqual(min_time, 3.0)
        self.assertAlmostEqual(left_time, 0.0)
        self.assertAlmostEqual(right_time, 6.0)

    def test_repair_ear_bounds_returns_none_when_annotation_outside_data(self) -> None:
        times = np.arange(5, dtype=float)
        data = np.ones(5)
        result = TimeSeriesViewer._repair_ear_bounds_from_trough_peak(
            times, data, annotation_onset=20.0, annotation_duration=5.0
        )
        self.assertIsNone(result)

    def test_ear_nth_peak_selects_second_and_third_peak(self) -> None:
        # min at index 0; peaks at indices 2, 5, 8 going right
        # data: 0, 1, 3, 2, 1, 4, 3, 2, 5, 4
        data = np.array([0.0, 1.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 5.0, 4.0])
        self.assertEqual(TimeSeriesViewer._ear_find_right_peak_index(data, 0, nth_peak=1), 2)
        self.assertEqual(TimeSeriesViewer._ear_find_right_peak_index(data, 0, nth_peak=2), 5)
        self.assertEqual(TimeSeriesViewer._ear_find_right_peak_index(data, 0, nth_peak=3), 8)

    def test_ear_nth_peak_returns_boundary_when_not_enough_peaks(self) -> None:
        # Only 1 peak to the right; asking for 3rd returns last index
        data = np.array([0.0, 1.0, 3.0, 2.0, 1.0])
        self.assertEqual(TimeSeriesViewer._ear_find_right_peak_index(data, 0, nth_peak=3), 4)


if __name__ == "__main__":
    unittest.main()
