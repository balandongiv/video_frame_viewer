"""SQLite-backed status tracking for discovered videos."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3
from typing import Iterable, List, Optional

STATUS_OPTIONS = ("pending", "ongoing", "complete")


@dataclass
class VideoStatusRecord:
    """Represents a persisted status entry for a single video segment."""

    video_path: Path
    subject_id: str
    recording_group_id: str
    segment_index: int
    status: str
    updated_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "VideoStatusRecord":
        return cls(
            video_path=Path(row["video_path"]),
            subject_id=row["subject_id"],
            recording_group_id=row["recording_group_id"],
            segment_index=int(row["segment_index"]),
            status=row["status"],
            updated_at=row["updated_at"],
        )


def normalize_subject_id(raw_subject: Optional[str]) -> str:
    """Normalize a subject identifier such as S01 into S1."""

    if not raw_subject:
        return "Unknown"

    digits = raw_subject[1:]
    try:
        numeric = int(digits)
    except ValueError:
        return raw_subject

    return f"S{numeric}"


def parse_video_metadata(path: Path) -> tuple[str, str, int]:
    """Parse the subject, recording group, and segment index from a video path."""

    stem = path.stem
    match = re.search(r"(S\d+_\d{8}_\d{6})(?:_(\d+))?", stem)
    if match:
        recording_group_id = match.group(1)
        segment_text = match.group(2)
        segment_index = int(segment_text) if segment_text else 1
        subject_match = re.match(r"(S\d+)", recording_group_id)
        raw_subject = subject_match.group(1) if subject_match else None
        subject_id = normalize_subject_id(raw_subject)
        return subject_id, recording_group_id, segment_index

    subject_match = re.search(r"S\d+", stem)
    raw_subject = subject_match.group(0) if subject_match else None
    subject_id = normalize_subject_id(raw_subject)
    return subject_id, stem, 1


class VideoStatusStore:
    """Persist and retrieve video status records using SQLite."""

    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path
        self.db_path = root_path / "video_status.db"
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS video_status (
                    video_path TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    recording_group_id TEXT NOT NULL,
                    segment_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        if self.connection:
            self.connection.close()

    def sync_with_videos(self, video_paths: Iterable[Path]) -> List[VideoStatusRecord]:
        """Ensure every discovered video has a status row and return records."""

        now = self._timestamp()
        paths = list(video_paths)
        with self.connection:
            for path in paths:
                subject_id, recording_group_id, segment_index = parse_video_metadata(path)
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO video_status
                        (video_path, subject_id, recording_group_id, segment_index, status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(path),
                        subject_id,
                        recording_group_id,
                        segment_index,
                        "pending",
                        now,
                    ),
                )
        return self.fetch_records(paths)

    def fetch_records(self, video_paths: Iterable[Path]) -> List[VideoStatusRecord]:
        records: List[VideoStatusRecord] = []
        for path in video_paths:
            row = self.connection.execute(
                """
                SELECT video_path, subject_id, recording_group_id, segment_index, status, updated_at
                FROM video_status
                WHERE video_path = ?
                """,
                (str(path),),
            ).fetchone()
            if row:
                records.append(VideoStatusRecord.from_row(row))
        return records

    def update_status_for_paths(self, paths: Iterable[Path], status: str) -> str:
        updated_at = self._timestamp()
        with self.connection:
            for path in paths:
                self.connection.execute(
                    """
                    UPDATE video_status
                    SET status = ?, updated_at = ?
                    WHERE video_path = ?
                    """,
                    (status, updated_at, str(path)),
                )
        return updated_at

    def update_status_for_group(self, recording_group_id: str, status: str) -> str:
        updated_at = self._timestamp()
        with self.connection:
            self.connection.execute(
                """
                UPDATE video_status
                SET status = ?, updated_at = ?
                WHERE recording_group_id = ?
                """,
                (status, updated_at, recording_group_id),
            )
        return updated_at

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
