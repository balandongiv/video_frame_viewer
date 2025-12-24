"""Test package initialization."""

from generate_mock_eog import VIDEO_PATH, regenerate_mock_eog

if VIDEO_PATH.exists():
    regenerate_mock_eog()
