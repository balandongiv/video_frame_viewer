# Agent Specification: Video Frame Navigation Tool (Python)

## 1. Purpose

Create a Python-based desktop application that allows users to:

- Scan a dataset directory for `.mov` video files.
- Select a video from the discovered list.
- Jump directly to a specific frame number.
- Navigate frames one by one or in fixed jumps.
- Apply a configurable frame offset (“shift”) when interpreting frame numbers.

The tool is intended for working with large video files (around 34 minutes long, ~900 MB each) in a drowsy driving dataset.

---

## 2. Target Environment and Technology

- **Programming language:** Python (user requirement).
- **Target OS:** Windows (directory examples use Windows-style paths).
- **Minimum Python version:** 3.8 or newer (unless constrained by chosen libraries).
- **Video handling:** Use a Python-compatible library that supports efficient random frame access for `.mov` files.
- **GUI framework:** Use any standard Python GUI toolkit (for example, Tkinter, PyQt, PySide, or similar). The choice should prioritize:
    - Ease of deployment on Windows.
    - Stable support for buttons, text inputs, labels, and list/tree views.

The agent may choose appropriate libraries that fit these constraints, as long as they are commonly used and reasonably documented.

---

## 3. Dataset and Directory Structure

- The primary dataset directory is under a root path such as:

    - `D:\dataset\drowsy_driving_raja\`

- Example video paths:

    - `D:\dataset\drowsy_driving_raja\S1\MD.mff.S01_20170519_043933.mov`
    - `D:\dataset\drowsy_driving_raja\S1\MD.mff.S01_20170519_043933_2.mov`

- General pattern (for understanding only, not strict parsing):

    - `D:\dataset\drowsy_driving_raja\S<subject_id>\MD.mff.<video_filename>.mov`

- Requirements for scanning:

    - Recursively scan all subfolders under a user-selected root directory.
    - Identify and list all files with the `.mov` extension.
    - Present the discovered videos in the UI so the user can select one to open.

---

## 4. Functional Requirements

### 4.1 Directory Selection and Scanning

- Provide a way for the user to:

    - Select a root directory (e.g., via a directory picker or a text field with a “Browse” button).
    - Trigger a scan that:
        - Recursively traverses all subfolders.
        - Finds all `.mov` files.
        - Populates a list or tree component with:
            - At minimum, the full file path or a combination of filename and relative path.

### 4.2 Video Selection

- Allow the user to select a single video from the discovered list.
- Once selected:
    - Load the video for frame-based navigation.
    - Initialize internal state (current frame, total frame count if available, etc.).

### 4.3 Frame Search and Jump

- Provide a numeric input field for entering a frame number (e.g., `100`).
- Provide a **Search** button that:

    - Reads the entered frame number.
    - Applies the configured frame shift (see Section 4.5).
    - Computes the effective frame index:
        - `effective_frame = user_input_frame + shift_frame_value`
    - Jumps the video to `effective_frame`, clamping to valid bounds if necessary.
    - Updates the displayed frame and the current frame indicator.

### 4.4 Frame Navigation Controls

For the currently loaded video, implement the following navigation buttons:

- **Left**
    - Move 1 frame backward from the current frame.
- **Right**
    - Move 1 frame forward from the current frame.
- **Left_Jump**
    - Move 10 frames backward from the current frame.
- **Right_Jump**
    - Move 10 frames forward from the current frame.

Additional behavior:

- After each navigation action:
    - Update the currently displayed frame.
    - Update the visible current frame number in the UI.

- Apply boundary checks:
    - Do not allow navigation to a frame index below the first frame.
    - Do not allow navigation past the last frame.
    - If an action would go out of bounds, clamp to the nearest valid frame.

### 4.5 Frame Shift (Offset) Feature

- Provide a numeric input for **Shift Frame** (frame offset).
- When the user sets a shift value:

    - Store this value as a global offset for frame interpretation.
    - Example: If the user enters a shift of `+10`, and then enters frame `100` in the main frame search, the actual frame used should be `110`.

- All frame searches via the **Search** button must use:

    - `effective_frame = input_frame + shift_frame_value`

- Shift value can be positive or negative, depending on the user’s needs to compensate for delays or offsets.

### 4.6 Current Frame Display

- Always display the current frame index in the UI.
- Ensure this value is updated when:
    - The user navigates with any of the frame buttons.
    - The user performs a frame search.
    - The video is initially loaded or reset.

---

## 5. Frame Handling and Indexing

- The application must work primarily with **frame indices**, not timestamps.
- Define a consistent frame index convention (for example, zero-based or one-based) and apply it consistently across:
    - Displayed frame numbers.
    - User input interpretation.
    - Internal calculations.
- For out-of-range requests (due to searches or navigation):

    - Clamp to the nearest valid frame.
    - Optionally show a status message or visual indication (for example, a label showing “Reached first frame” or “Reached last frame”).

---

## 6. Performance and Robustness

- Video files are large (~900 MB, ~34 minutes each), so:

    - Use a video library that supports efficient random access to individual frames without loading the entire file into memory.
    - Ensure that frame-seeking operations do not cause excessive delays or UI freezes. If necessary, use appropriate threading or asynchronous patterns provided by the chosen GUI framework.

- Handle common error conditions gracefully:

    - Missing or unreadable files.
    - Unsupported video codecs.
    - Invalid user input (non-numeric frame values, empty fields, etc.).

- Provide minimal but clear feedback when errors occur (e.g., a status bar message or dialog).

---

## 7. User Interface Summary

The UI must include at least:

1. **Directory Controls**
    - Field and/or button to select the root directory.
    - Button to start scanning for `.mov` files.

2. **Video List / Tree**
    - Displays all discovered `.mov` files.
    - Allows selecting one video for viewing.

3. **Frame Search Controls**
    - Numeric input for frame number.
    - **Search** button to jump to the specified frame (after applying the shift).

4. **Navigation Buttons**
    - **Left**: −1 frame
    - **Right**: +1 frame
    - **Left_Jump**: −10 frames
    - **Right_Jump**: +10 frames

5. **Shift Frame Controls**
    - Numeric input for shift frame value.
    - Button or mechanism to apply/update the shift value.

6. **Status Display**
    - Label or field showing the current frame index.
    - Optional area for short status messages (e.g., scan completed, errors, out-of-range frame request).

---

## 8. Assumptions and Clarifications

- The application is intended for local, offline use on a Windows machine.
- Only `.mov` files need to be considered during scanning.
- The agent may choose suitable Python libraries for:
    - Video decoding and frame access.
    - GUI construction and event handling.
- No advanced video editing, saving, or annotation features are required in this version—only viewing and frame navigation.

---

## 9. Acceptance Criteria

The implementation should be considered complete when:

1. The user can choose a root directory and the application successfully finds all `.mov` files recursively.
2. The user can select any discovered video and see its frames displayed.
3. The user can:
    - Enter a frame number, press **Search**, and see the video jump to that frame (respecting the shift value).
    - Use **Left**, **Right**, **Left_Jump**, and **Right_Jump** to navigate frames, with correct boundary handling.
4. The **Shift Frame** value is applied correctly to all subsequent frame searches.
5. The current frame index is always shown and kept in sync with the displayed frame.
6. The application runs with acceptable responsiveness given the large video sizes, without crashing on normal usage.

