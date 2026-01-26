"""User entry point: uses config on Desktop."""
import sys
from pathlib import Path

# Add src to path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from cli import main
from config import AppConfig, save_config

def get_desktop_path():
    return Path.home() / "Desktop"

def prompt_for_path(prompt_text):
    while True:
        path_str = input(prompt_text).strip()
        if path_str:
            return Path(path_str)
        print("Path is required.")

def main_user():
    desktop = get_desktop_path()
    config_path = desktop / "config_video_frame_viewer.yaml"

    if not config_path.exists():
        print(f"Configuration file not found at {config_path}")
        print("Please provide the following paths to set up your environment.")
        
        dataset_root = prompt_for_path("Enter the dataset root folder (where videos are): ")
        csv_dir = prompt_for_path("Enter the folder containing processed CSV files: ")
        fif_dir = prompt_for_path("Enter the folder containing FIF files (often same as CSV folder): ")

        # Create config
        # If user provides absolute paths, they will be used directly.
        config = AppConfig(
            dataset_root=dataset_root,
            csv_dir=str(csv_dir),
            fif_dir=str(fif_dir),
            ui={"ear_baseline": 0.0}
        )
        
        save_config(config, config_path)
        print(f"Configuration saved to {config_path}")

    # Run app with this config
    sys.exit(main(["--config", str(config_path)]))

if __name__ == "__main__":
    main_user()
