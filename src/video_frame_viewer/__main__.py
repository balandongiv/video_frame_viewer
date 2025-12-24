"""Enable ``python -m video_frame_viewer`` execution."""
from video_frame_viewer.cli import main

if __name__ == "__main__":  # pragma: no cover - convenience entrypoint
    raise SystemExit(main())
