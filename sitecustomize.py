# Auto-loaded by Python at startup if present on sys.path.
# We use it to ensure required Windows DLL paths are configured in every process
# (including multiprocessing children) before importing native extensions like `pystk2`.
from __future__ import annotations

import os
import sys
from pathlib import Path


def _add_dir(p: Path) -> None:
    try:
        if not p.is_dir():
            return
        try:
            os.add_dll_directory(str(p))  # ensure current process can resolve DLLs
        except Exception:
            pass
        # Prepend to PATH so any further child processes also inherit
        path_sep = ";"
        env_path = os.environ.get("PATH", "")
        parts = env_path.split(path_sep) if env_path else []
        if all(Path(x) != p for x in map(Path, parts)):
            os.environ["PATH"] = str(p) + (path_sep + env_path if env_path else "")
    except Exception:
        # Never fail process startup due to path setup
        pass


if os.name == "nt":
    # Conda env DLLs (libpng, zlib, etc.)
    env_bin = Path(sys.prefix) / "Library" / "bin"
    _add_dir(env_bin)

    # SuperTuxKart common install locations (Program Files; any version folder)
    pf = Path(r"C:\\Program Files")
    pf86 = Path(r"C:\\Program Files (x86)")
    for base in (pf, pf86):
        try:
            for d in base.glob("SuperTuxKart*"):
                _add_dir(d)
                _add_dir(d / "bin")
        except Exception:
            pass
    # Steam default path
    _add_dir(pf86 / "Steam" / "steamapps" / "common" / "SuperTuxKart" / "bin")
