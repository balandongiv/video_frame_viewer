# Video Frame Viewer

An interactive desktop tool for browsing drowsy-driving study videos alongside synchronized EEG/EOG/EAR signals and CSV annotations. The viewer scans a dataset tree for `.mov` recordings, lets you jump directly to frames, overlays time-series signals, and edits or nudges annotations in context.

## Features

- Discover `.mov` recordings under a configurable dataset root.
- Frame navigation with direct frame search, ±1/±10 frame jumps, zoom controls, and frame-offset (“shift”) support.
- Time-series overlay for EEG/EOG/EAR signals with channel selection, EAR gain/ear baseline controls, and annotation lanes.
- Annotation editing: add regions with **Ctrl + left drag**, edit/delete via context menu, save to CSV, and single/bulk nudge controls.
- **Session Persistence:** Remembers your frame shift and last stop position per video/dataset.
- **Two Run Modes:** dedicated entry points for developers (`main_debug.py`) and users (`main_user.py`).

## Installation

- Python **3.9+** is required.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## How to Run

### 1. Developer Mode (`main_debug.py`)

Intended for maintainers or anyone running inside the repository with the default directory structure.

```bash
python main_debug.py
```

- **Configuration:** Always uses `config.dev.yaml` from the repository root.
- **Paths:** Expects relative paths (e.g., `./test_data/...`) as defined in the repo.

### 2. User Mode (`main_user.py`)

Intended for general use on any machine, supporting absolute paths (e.g., `D:\dataset\...`).

```bash
python main_user.py
```

- **Configuration:** Looks for `config_video_frame_viewer.yaml` on your **Desktop**.
- **First Run:** If the config file is missing, the app will prompt you to enter:
  1.  **Dataset Root:** The folder containing your videos (e.g., `D:\Data\Videos`).
  2.  **CSV Directory:** The folder containing processed `.csv` files (e.g., `D:\Data\Processed`).
  3.  **FIF Directory:** The folder containing `.fif` files (often the same as CSV).
- **Subsequent Runs:** Automatically loads settings from the Desktop config file.

## Configuration on a New Machine

If your dataset paths differ from the repository defaults (common on different computers or OSes):

1.  Run `python main_user.py`.
2.  Follow the prompts to enter your local paths.
    *   **Example (Windows):**
        *   Dataset Root: `C:\Users\Name\Documents\StudyData`
        *   CSV/FIF Directory: `C:\Users\Name\Documents\StudyData_Processed`
3.  The app creates `config_video_frame_viewer.yaml` on your Desktop. You can edit this file manually later if needed.

## Session Persistence

The application automatically saves your progress for each video context to a file named `VideoFrameViewers.yaml`.

- **Location:** Created in the same folder as the loaded `ear_eog.csv` (next to your data).
- **What is saved:**
  - **Shift Frame:** The synchronization offset you set in the UI.
  - **Stop Position:** The last frame you were viewing when you closed the app or switched videos.
- **Behavior:**
  - On loading a video, the app checks for this file.
  - If found, it restores your shift value and jumps to your last stop position.
  - If not found, it starts with default settings (frame 0, shift 0).

## Keyboard Shortcuts & Controls

### Navigation
| Key | Action |
| :--- | :--- |
| **Left Arrow** | Jump backward (default 10 frames) |
| **Right Arrow** | Jump forward (default 10 frames) |
| **Ctrl + Left** | Step backward 1 frame |
| **Ctrl + Right** | Step forward 1 frame |
| **[** or **P** | Jump to **Previous** annotation |
| **]** or **N** | Jump to **Next** annotation |
| **Ctrl + N** | Jump to **Next** annotation & center on **EAR minimum** |

### Annotations
| Action | Control |
| :--- | :--- |
| **Create Annotation** | **Ctrl + Left Click & Drag** on the time-series plot |
| **Edit/Delete** | **Right Click** on an annotation region |
| **Save Changes** | **Ctrl + S** (saves to `ear_eog.csv`) |

### Zoom & View
| Control | Action |
| :--- | :--- |
| **Ctrl + Mouse Wheel** | Zoom time-series in/out (centered on mouse cursor) |
| **Click & Drag** | Pan the video frame (when zoomed in) |

## Development

- **Source Code:** All source files are located in `src/`.
- **Linting:** `ruff check .`
- **Testing:** `pytest`