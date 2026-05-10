import unittest
from pathlib import Path

from cao_2018_gui import Cao2018Viewer, CaoRecording


class _FakeILoc:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, index):
        return self._rows[index]


class _FakeDataFrame:
    def __init__(self, rows):
        self._rows = [dict(row) for row in rows]
        self.columns = list(self._rows[0].keys()) if self._rows else []
        self.index = list(range(len(self._rows)))
        self.iloc = _FakeILoc(self._rows)

    def iterrows(self):
        for index, row in enumerate(self._rows):
            yield index, row


class CaoBlinkerParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = Cao2018Viewer.__new__(Cao2018Viewer)

    def test_parses_nested_channel_frames_blinkfits_payload(self) -> None:
        payload = {
            "requested_channels": ["FP1", "FP2"],
            "channels": {
                "FP1": {
                    "requested_channel": "FP1",
                    "frames": {
                        "blinkFits": _FakeDataFrame(
                            [
                                {"leftBase": 450.0, "rightBase": 500.0, "maxFrame": 470.0},
                                {"leftBase": 200.0, "rightBase": 260.0, "maxFrame": 230.0},
                            ]
                        ),
                        "blinkStats": _FakeDataFrame([{"srate": 100.0}]),
                    },
                }
            },
        }

        rows = self.viewer._annotations_from_blinker_payload(payload)

        self.assertEqual(
            rows,
            [
                {"onset": "2.000000", "duration": "0.600000", "description": "eye_blink"},
                {"onset": "4.500000", "duration": "0.500000", "description": "eye_blink"},
            ],
        )

    def test_parses_direct_onset_duration_table(self) -> None:
        payload = _FakeDataFrame(
            [
                {"onset": 3.2, "duration": 0.4},
                {"onset": 1.0, "duration": 0.2},
            ]
        )

        rows = self.viewer._annotations_from_blinker_payload(payload)

        self.assertEqual(
            rows,
            [
                {"onset": "1.000000", "duration": "0.200000", "description": "eye_blink"},
                {"onset": "3.200000", "duration": "0.400000", "description": "eye_blink"},
            ],
        )

    def test_recording_list_uses_subject_and_session_only(self) -> None:
        recording = CaoRecording(
            subject_id="S01",
            session_id="051017m",
            folder=Path("/tmp/S01/051017m"),
            ts_path=Path("/tmp/S01/051017m/example_ds20hz.fif"),
        )

        self.assertEqual(self.viewer._recording_list_text(recording), "S01/051017m")


if __name__ == "__main__":
    unittest.main()
