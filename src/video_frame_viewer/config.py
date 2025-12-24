"""Configuration management for the Video Frame Viewer."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from platformdirs import user_config_path

LOGGER = logging.getLogger(__name__)

APP_NAME = "video-frame-viewer"
ENV_CONFIG_PATH = "DROWSY_CONFIG"
ENV_DATASET_ROOT = "DROWSY_DATASET_ROOT"
DEFAULT_CONFIG_FILENAME = "config.yaml"
DEFAULT_DEV_CONFIG_NAME = "config.dev.yaml"


class ConfigNotFoundError(FileNotFoundError):
    """Raised when configuration could not be resolved without prompting."""


def _expand_path(raw_path: str, relative_to: Optional[Path] = None) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and relative_to:
        path = (relative_to / path).resolve()
    return path


def _default_processed_root(dataset_root: Path) -> Path:
    return dataset_root.parent / f"{dataset_root.name}_processed"


def _user_config_path() -> Path:
    return user_config_path(APP_NAME, ensure_exists=True) / DEFAULT_CONFIG_FILENAME


@dataclass
class AppConfig:
    """Runtime configuration for locating datasets and storing UI preferences."""

    dataset_root: Path
    mov_dir: Optional[str] = None
    csv_dir: Optional[str] = None
    fif_dir: Optional[str] = None
    ui: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            **({"mov_dir": self.mov_dir} if self.mov_dir else {}),
            **({"csv_dir": self.csv_dir} if self.csv_dir else {}),
            **({"fif_dir": self.fif_dir} if self.fif_dir else {}),
            **({"ui": self.ui} if self.ui else {}),
        }

    @property
    def video_root(self) -> Path:
        if self.mov_dir:
            return _expand_path(self.mov_dir, self.dataset_root)
        return self.dataset_root

    @property
    def time_series_root(self) -> Path:
        if self.fif_dir:
            return _expand_path(self.fif_dir, self.dataset_root)
        return _default_processed_root(self.dataset_root)

    @property
    def annotation_root(self) -> Path:
        if self.csv_dir:
            return _expand_path(self.csv_dir, self.dataset_root)
        return self.time_series_root


@dataclass
class ConfigResolution:
    """Result of resolving configuration inputs."""

    config: AppConfig
    path: Optional[Path]
    source: str


def _load_yaml_config(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if "dataset_root" not in data:
        raise ValueError(f"Config file at {path} missing required dataset_root")
    dataset_root = _expand_path(str(data["dataset_root"]), path.parent)
    return AppConfig(
        dataset_root=dataset_root,
        mov_dir=data.get("mov_dir"),
        csv_dir=data.get("csv_dir"),
        fif_dir=data.get("fif_dir"),
        ui=data.get("ui") or {},
    )


def save_config(config: AppConfig, path: Optional[Path] = None) -> Path:
    target = path or _user_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
    LOGGER.info("Saved configuration to %s", target)
    return target


def _find_repo_config() -> Optional[Path]:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / DEFAULT_DEV_CONFIG_NAME
        if candidate.exists():
            repo_marker = parent / ".git"
            if repo_marker.exists():
                return candidate
    return None


def _prompt_for_dataset_root() -> Path:
    while True:
        user_input = input("Enter the dataset root path: ").strip()
        if user_input:
            return Path(user_input).expanduser().resolve()
        print("Dataset root is required. Please enter a valid path.")


def resolve_config(
    cli_config: Optional[Path] = None,
    allow_prompt: bool = True,
) -> ConfigResolution:
    if cli_config:
        config_path = cli_config.expanduser()
        return ConfigResolution(_load_yaml_config(config_path), config_path, "cli")

    env_path = os.environ.get(ENV_CONFIG_PATH)
    if env_path:
        config_path = Path(env_path).expanduser()
        return ConfigResolution(_load_yaml_config(config_path), config_path, "env_config")

    env_root = os.environ.get(ENV_DATASET_ROOT)
    if env_root:
        config = AppConfig(dataset_root=Path(env_root).expanduser())
        return ConfigResolution(config, None, "env_dataset_root")

    user_path = _user_config_path()
    if user_path.exists():
        return ConfigResolution(_load_yaml_config(user_path), user_path, "user")

    repo_config = _find_repo_config()
    if repo_config:
        return ConfigResolution(_load_yaml_config(repo_config), repo_config, "repo")

    if not allow_prompt:
        raise ConfigNotFoundError("No configuration found in CLI args, env vars, or defaults.")

    dataset_root = _prompt_for_dataset_root()
    config = AppConfig(dataset_root=dataset_root)
    saved_path = save_config(config, user_path)
    return ConfigResolution(config, saved_path, "user_created")
