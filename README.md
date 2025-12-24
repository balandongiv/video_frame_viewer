# Video Frame Viewer

An interactive desktop tool for browsing drowsy-driving study videos alongside synchronized EEG/EOG/EAR signals and CSV annotations. The viewer scans a dataset tree for `.mov` recordings, lets you jump directly to frames, overlays time-series signals, and edits or nudges annotations in context.

## Features

- Discover `.mov` recordings under a configurable dataset root (MD.mff naming supported).
- Frame navigation with direct frame search, ±1/±10 frame jumps, zoom controls, and frame-offset (“shift”) support.
- Time-series overlay for EEG/EOG/EAR signals with channel selection, EAR gain/ear baseline controls, and annotation lanes.
- Annotation editing: add regions with **Ctrl + left drag**, edit/delete via context menu, save to CSV, and single/bulk nudge controls.
- Keyboard shortcuts: arrow keys (frame jumps), **Ctrl + arrow** (single-step), `[`/`]` or **P/N** (jump between annotations), **Ctrl + S** (save annotations).
- Debug toggle for bundled test fixtures when running from a git checkout.

## Installation

- Python **3.9+** is required.
- Install directly from GitHub:

  ```bash
  pip install "git+https://github.com/<your-org>/video_frame_viewer.git"
  ```

- For development (recommended for run-from-repo debugging):

  ```bash
  pip install -e ".[dev]"
  ```

  This installs the package in editable mode plus `ruff`/`pytest` for linting and tests.

## Quickstart

1. Run the CLI entry point:

   ```bash
   drowsy-viewer
   ```

   Use `--verbose` for additional logging, or `--config <path>` to point to a specific YAML config.

2. On first launch (when no config exists), you’ll be prompted for the **dataset root**. The tool saves a YAML config automatically and reuses it next time.

3. The UI shows the dataset root input, a video list, and the frame/time-series panes. Select a video to load frames and synchronized signals.

4. Update the default dataset root later with:

   ```bash
   drowsy-viewer --set-root /path/to/dataset
   ```

5. Advanced overrides:
   - `DROWSY_CONFIG`: path to a YAML config file.
   - `DROWSY_DATASET_ROOT`: override the dataset root without editing files.
   - `--config <path>`: highest precedence for config files.

### Config file locations

- **Windows:** `%APPDATA%\video-frame-viewer\config.yaml` (or `%LOCALAPPDATA%`)
- **macOS:** `~/Library/Application Support/video-frame-viewer/config.yaml`
- **Linux:** `~/.config/video-frame-viewer/config.yaml`

### Config precedence

1. `--config <path>`
2. `DROWSY_CONFIG` or `DROWSY_DATASET_ROOT`
3. User config in the OS-specific path above
4. `config.dev.yaml` in a git checkout (for run-from-repo defaults)
5. Interactive prompt to create a user config

## Expected dataset layout

```
<dataset_root>/
├─ S01/
│  ├─ MD.mff.S01_20170519_043933.mov
│  ├─ MD.mff.S01_20170519_043933_2.mov   # additional segments
│  └─ ...
├─ S02/
│  └─ MD.mff.S02_20170519_053933.mov
└─ ...

<dataset_root>_processed/                # default time-series + annotation root
├─ S01/
│  └─ S01_20170519_043933/
│     ├─ ear_eog.fif                     # EEG/EOG/EAR signals
│     └─ ear_eog.csv                     # annotations (created/updated by the app)
└─ ...
```

- If your processed FIF/CSV files live elsewhere, set `fif_dir`/`csv_dir` relative to `dataset_root` in the YAML config.
- Videos can also reside in a subfolder when `mov_dir` is set in the config.

## Configuration

Example `config.yaml`:

```yaml
dataset_root: /data/drowsy_driving_raja
mov_dir: videos                   # optional, relative to dataset_root
fif_dir: ../drowsy_driving_raja_processed
csv_dir: ../drowsy_driving_raja_processed
ui:
  ear_baseline: 0.0               # optional UI defaults
```

- `dataset_root` **(required)**: root that contains the subject folders.
- `mov_dir`, `fif_dir`, `csv_dir` **(optional)**: override locations relative to `dataset_root`.
- `ui` **(optional)**: UI defaults (e.g., EAR baseline).

## Usage guide

- **Scanning videos:** Enter/select the dataset root and click **Rescan**. The app searches for MD.mff `.mov` files under the configured video root.
- **Frame navigation:** Jump to a frame number, use ±1 frame buttons, or ±10-frame jumps. Zoom controls resize the displayed frame.
- **Frame shift & sync:** Set a frame shift or sync offset (in seconds); both affect frame-to-time alignment and annotation jumps.
- **Signals & channels:** The time-series pane loads the associated `ear_eog.fif` file. Toggle channels, boost `EAR-avg_ear`, and set the red EAR baseline.
- **Annotations:**
  - **Add:** Hold **Ctrl** and left-drag to create a region.
  - **Edit/delete:** Right-click a region for the context menu.
  - **Save:** **Ctrl + S** or the Save button writes `ear_eog.csv` under the annotation root.
  - **Nudge:** Use single/bulk nudge controls (in frames) to shift annotations.
- **Keyboard shortcuts:** arrow keys for jumps, **Ctrl + arrow** for single steps, `[`/`]` or **P/N** to move between annotations, **Ctrl + S** to save.
- **Debug data:** Enable “Use test data” to work with the bundled `test_data` fixtures via `config.dev.yaml` when running from a repo checkout.

## Troubleshooting

- **Dataset root not found:** Confirm the path in your config or the UI. Update via `--set-root` or edit the YAML.
- **No videos discovered:** Verify `mov_dir` and the presence of `.mov` files (MD.mff naming). Check the “Video directory not found” status for the resolved path.
- **Missing FIF/CSV:** The app reports the expected path; adjust `fif_dir`/`csv_dir` or ensure processed data is present.
- **Annotation save issues:** Ensure you have write permissions to the annotation directory. The app writes a temporary file then atomically replaces the target.
- **Headless environments:** Qt may require an offscreen platform plugin; set `QT_QPA_PLATFORM=offscreen` if needed.

## Development

- Install dev dependencies: `pip install -e ".[dev]"`
- Lint: `ruff check .`
- Tests: `pytest`
- Run locally from the repo:
  - `python main.py` (adds `src/` to `PYTHONPATH` for convenience), or
  - `python -m video_frame_viewer.cli`
- The repo ships `config.dev.yaml` pointing to `test_data/` for predictable run-from-repo behavior.

## License

This project is distributed under the repository’s license (see accompanying files or contact the maintainers for details).
