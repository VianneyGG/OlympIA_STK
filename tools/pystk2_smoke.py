from __future__ import annotations

import logging

import os
import sys
from pathlib import Path

# Ensure we can import project modules from src/
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from env_utils import ensure_windows_dlls


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    ensure_windows_dlls()
    import pystk2

    logging.info("pystk2 version: %s", getattr(pystk2, "__version__", "unknown"))
    logging.info("Initializing PySuperTuxKart2 (no graphics)...")
    pystk2.init(pystk2.GraphicsConfig.none())
    logging.info("Listing tracks...")
    tracks = pystk2.list_tracks(pystk2.RaceConfig.RaceMode.NORMAL_RACE)
    logging.info("Found %d tracks, e.g.: %s", len(tracks), tracks[:5])


if __name__ == "__main__":
    main()
