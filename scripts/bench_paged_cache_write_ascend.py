#!/usr/bin/env python3
"""Benchmark paged cache write on one selected NPU without locking clocks."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from benchmarks.ascend.paged_cache_write import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
