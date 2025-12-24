from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from video_frame_viewer.config import (
    ENV_CONFIG_PATH,
    ENV_DATASET_ROOT,
    AppConfig,
    ConfigNotFoundError,
    resolve_config,
    save_config,
)


class ResolveConfigTests(TestCase):
    def test_cli_config_overrides_other_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            dataset_root = base / "data"
            dataset_root.mkdir()
            config_path = base / "config.yaml"
            save_config(AppConfig(dataset_root=dataset_root), config_path)

            with patch.dict(os.environ, {}, clear=True):
                resolution = resolve_config(cli_config=config_path, allow_prompt=False)

            self.assertEqual(resolution.source, "cli")
            self.assertEqual(resolution.path, config_path)
            self.assertEqual(resolution.config.dataset_root, dataset_root)

    def test_env_dataset_root_used_when_no_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_root = Path(tmp_dir) / "data"
            with patch.dict(
                os.environ, {ENV_DATASET_ROOT: str(dataset_root), ENV_CONFIG_PATH: ""}, clear=True
            ):
                resolution = resolve_config(allow_prompt=False)

            self.assertEqual(resolution.source, "env_dataset_root")
            self.assertEqual(resolution.config.dataset_root, dataset_root)
            self.assertIsNone(resolution.path)

    def test_missing_configuration_raises_without_prompt(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch(
            "video_frame_viewer.config._find_repo_config", return_value=None
        ):
            with self.assertRaises(ConfigNotFoundError):
                resolve_config(allow_prompt=False)

    def test_relative_dataset_root_resolves_against_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            config_path = base / "config.yaml"
            config_path.write_text("dataset_root: ./nested/data\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                resolution = resolve_config(cli_config=config_path, allow_prompt=False)

            expected_root = (config_path.parent / "nested" / "data").resolve()
            self.assertEqual(resolution.config.dataset_root, expected_root)
