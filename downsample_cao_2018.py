"""Batch-downsample all FIF files in the Cao 2018 sustained-attention driving dataset.

For each raw FIF found under the dataset root (two-level S{subject}/{session}/),
applies a 10 Hz lowpass FIR filter, resamples to 20 Hz, and saves a cache FIF
file next to the original.  The GUI (cao_2018_main.py) will prefer these cache
files over the full-resolution FIF when displaying signals.

Usage
-----
    python downsample_cao_2018.py
    python downsample_cao_2018.py --dataset-root D:\\dataset\\sustained_attention_driving
    python downsample_cao_2018.py --force          # re-create existing caches
    python downsample_cao_2018.py --dry-run        # show what would be done
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mne

LOWPASS_HZ: float = 10.0
TARGET_SFREQ: float = 20.0
CACHE_SUFFIX: str = "_ds20hz.fif"
DEFAULT_ROOT: Path = Path(r"D:\dataset\sustained_attention_driving")


def _discover_fifs(root: Path) -> list[Path]:
    """Find all raw (non-epoch, non-cache) FIF files under the two-level folder tree."""
    results = []
    for fif in sorted(root.rglob("*.fif"), key=lambda p: p.as_posix().lower()):
        if fif.name.endswith(CACHE_SUFFIX):
            continue
        if "-epo.fif" in fif.name:
            continue
        results.append(fif)
    return results


def _cache_path(fif_path: Path) -> Path:
    return fif_path.with_name(fif_path.stem + CACHE_SUFFIX)


def _downsample(fif_path: Path, force: bool, dry_run: bool) -> str:
    """Process one FIF. Returns a one-line status string."""
    cache = _cache_path(fif_path)

    if cache.exists() and not force:
        return f"  SKIP  {fif_path.name}  (cache exists)"

    if dry_run:
        action = "RE-CREATE" if cache.exists() else "CREATE"
        return f"  {action}  {cache}"

    t0 = time.monotonic()
    raw = mne.io.read_raw_fif(str(fif_path), preload=True, verbose="ERROR")
    original_sfreq = raw.info["sfreq"]
    raw.filter(None, LOWPASS_HZ, fir_design="firwin", verbose="ERROR")
    raw.resample(TARGET_SFREQ, verbose="ERROR")
    raw.save(str(cache), overwrite=True, verbose="ERROR")
    elapsed = time.monotonic() - t0
    return (
        f"  OK    {fif_path.name}"
        f"  ({original_sfreq:.0f} Hz -> {TARGET_SFREQ:.0f} Hz, {elapsed:.1f}s)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-downsample Cao 2018 FIF files to 20 Hz."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Cao 2018 dataset root. Default: {DEFAULT_ROOT}",
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

    fifs = _discover_fifs(root)
    if not fifs:
        print(f"No raw FIF files found under {root}.")
        return 0

    label = "DRY RUN — " if args.dry_run else ""
    print(
        f"{label}Found {len(fifs)} FIF file(s) under {root}\n"
        f"Lowpass: {LOWPASS_HZ:.0f} Hz  |  Target sfreq: {TARGET_SFREQ:.0f} Hz  |  "
        f"Cache suffix: {CACHE_SUFFIX}\n"
    )

    ok = skip = fail = 0
    for i, fif in enumerate(fifs, 1):
        rel = fif.relative_to(root)
        print(f"[{i}/{len(fifs)}] {rel}")
        try:
            status = _downsample(fif, force=args.force, dry_run=args.dry_run)
            print(status)
            if "SKIP" in status:
                skip += 1
            else:
                ok += 1
        except Exception as exc:
            print(f"  FAIL  {exc}", file=sys.stderr)
            fail += 1

    print(f"\nDone.  processed={ok}  skipped={skip}  failed={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
