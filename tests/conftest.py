"""Test configuration and path setup."""
import sys
from pathlib import Path

# Add src to python path for testing
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))
