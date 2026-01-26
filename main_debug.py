"""Developer entry point: always uses config.dev.yaml from the repository."""
import sys
from pathlib import Path

# Add src to path
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from cli import main

if __name__ == "__main__":
    # Force loading config.dev.yaml
    sys.exit(main(["--config", "config.dev.yaml"]))
