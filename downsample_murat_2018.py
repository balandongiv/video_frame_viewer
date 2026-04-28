"""Batch-downsample all EDF files in the Murat 2018 dataset.

For each EDF found under the dataset root, applies a 20 Hz lowpass FIR filter,
resamples to 40 Hz, and saves a cache FIF file next to the original.  The GUI
(murat_2018_main.py) will use these cache files instead of the full-resolution
EDF when displaying signals.

Usage
-----
    python downsample_murat_2018.py
    python downsample_murat_2018.py --dataset-root D:\\dataset\\murat_2018
    python downsample_murat_2018.py --force          # re-create existing caches
    python downsample_murat_2018.py --dry-run        # show what would be done
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mne

LOWPASS_HZ: float = 20.0
TARGET_SFREQ: float = 40.0
CACHE_SUFFIX: str = "_ds20hz.fif"
DEFAULT_ROOT: Path = Path(r"D:\dataset\murat_2018")


def _discover_edfs(root: Path) -> list[Path]:
    return sorted(root.rglob("*.edf"), key=lambda p: p.as_posix().lower())


def _cache_path(edf_path: Path) -> Path:
    return edf_path.with_name(edf_path.stem + CACHE_SUFFIX)


def _downsample(edf_path: Path, force: bool, dry_run: bool) -> str:
    """Process one EDF. Returns a one-line status string."""
    cache = _cache_path(edf_path)

    if cache.exists() and not force:
        return f"  SKIP  {edf_path.name}  (cache exists)"

    if dry_run:
        action = "RE-CREATE" if cache.exists() else "CREATE"
        return f"  {action}  {cache}"

    t0 = time.monotonic()
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    original_sfreq = raw.info["sfreq"]
    raw.filter(None, LOWPASS_HZ, fir_design="firwin", verbose="ERROR")
    raw.resample(TARGET_SFREQ, verbose="ERROR")
    raw.save(str(cache), overwrite=True, verbose="ERROR")
    elapsed = time.monotonic() - t0
    return (
        f"  OK    {edf_path.name}"
        f"  ({original_sfreq:.0f} Hz -> {TARGET_SFREQ:.0f} Hz, {elapsed:.1f}s)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-downsample Murat 2018 EDF files.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Murat 2018 dataset root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-create cache files even if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing any files.",
    )
    args = parser.parse_args(argv)

    root: Path = args.dataset_root.expanduser().resolve()
    if not root.exists():
        print(f"ERROR: dataset root not found: {root}", file=sys.stderr)
        return 1

    edfs = _discover_edfs(root)
    if not edfs:
        print(f"No EDF files found under {root}.")
        return 0

    label = "DRY RUN — " if args.dry_run else ""
    print(
        f"{label}Found {len(edfs)} EDF file(s) under {root}\n"
        f"Lowpass: {LOWPASS_HZ:.0f} Hz  |  Target sfreq: {TARGET_SFREQ:.0f} Hz  |  "
        f"Cache suffix: {CACHE_SUFFIX}\n"
    )

    ok = skip = fail = 0
    for i, edf in enumerate(edfs, 1):
        print(f"[{i}/{len(edfs)}] {edf.parent.name}/{edf.name}")
        try:
            status = _downsample(edf, force=args.force, dry_run=args.dry_run)
            print(status)
            if "SKIP" in status:
                skip += 1
            else:
                ok += 1
        except Exception as exc:
            print(f"  FAIL  {exc}", file=sys.stderr)
            fail += 1

    print(
        f"\nDone.  processed={ok}  skipped={skip}  failed={fail}"
    )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
