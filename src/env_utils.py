"""
Environment utilities for SuperTuxKart Gymnasium (pystk2_gymnasium).

- Windows DLL search path setup (conda `Library/bin`, SuperTuxKart installs)
- Tarfile compatibility patch for Path-like members (Python 3.9+)
- Env factory with optional reward shaping and VecNormalize stats loading
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

import gymnasium as gym
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Optional, only used for type hints and default agent
    from pystk2_gymnasium import AgentSpec
except Exception:  # pragma: no cover - type-only import fallback
    AgentSpec = object  # type: ignore


def ensure_windows_dlls() -> None:
    """Ensure Windows can resolve dependent DLLs for `pystk2` in parent and children.

    We both:
    - call `os.add_dll_directory(...)` for the current process, and
    - prepend the same locations to the `PATH` env var so spawned processes inherit it.

    Searched locations:
    - Conda env DLLs: `<sys.prefix>/Library/bin`
    - SuperTuxKart installs (common):
      - `C:\\Program Files\\SuperTuxKart 1.4\\bin` and parent folder
      - `C:\\Program Files\\SuperTuxKart\\bin` and parent folder
      - Steam: `C:\\Program Files (x86)\\Steam\\steamapps\\common\\SuperTuxKart\\bin`

    Safe to call multiple times and on non-Windows.
    """
    if os.name != "nt":
        return

    def _add_dir(p: Path) -> None:
        try:
            if not p.is_dir():
                return
            # Make available in current process
            try:
                os.add_dll_directory(str(p))
            except Exception:
                # Older Pythons or failures — ignore
                pass
            # Prepend to PATH so child processes inherit
            path_sep = ";"
            env_path = os.environ.get("PATH", "")
            paths = env_path.split(path_sep) if env_path else []
            norm = str(p)
            if all(Path(x) != p for x in map(Path, paths)):
                os.environ["PATH"] = norm + (path_sep + env_path if env_path else "")
        except Exception:
            # Be defensive: never crash on path setup
            pass

    # Conda DLLs (zlib, libpng, etc.)
    env_bin = Path(sys.prefix) / "Library" / "bin"
    _add_dir(env_bin)

    # SuperTuxKart typical install locations (glob across versions)
    pf = Path(r"C:\\Program Files")
    pf86 = Path(r"C:\\Program Files (x86)")
    stk_candidates: list[Path] = []
    for base in (pf, pf86):
        try:
            for d in base.glob("SuperTuxKart*"):
                stk_candidates.append(d)
                stk_candidates.append(d / "bin")
        except Exception:
            pass
    # Common Steam location
    stk_candidates.append(pf86 / "Steam" / "steamapps" / "common" / "SuperTuxKart" / "bin")
    for d in stk_candidates:
        _add_dir(d)

    # Ensure child Python processes load our sitecustomize.py (located at repo root)
    # so they also call AddDllDirectory on startup before importing native modules.
    try:
        root = Path(__file__).resolve().parents[1]
        if (root / "sitecustomize.py").is_file():
            sep = ";"
            py_path = os.environ.get("PYTHONPATH", "")
            parts = py_path.split(sep) if py_path else []
            if all(Path(p) != root for p in map(Path, parts)):
                os.environ["PYTHONPATH"] = str(root) + (sep + py_path if py_path else "")
    except Exception:
        pass


def _patch_tarfile_pathlike() -> None:
    """Patch tarfile to tolerate Path-like member names (Python 3.9+ behavior).

    Some third-party packages may pass pathlib.Path as tar member names, which breaks
    stdlib internals expecting strings. This patch coerces `member.name` to `str`.
    """
    import tarfile

    _orig_get_filtered_attrs = getattr(tarfile, "_get_filtered_attrs", None)
    if _orig_get_filtered_attrs is None:
        return

    def _patched_get_filtered_attrs(member, targetpath, tarinfo_isdir=False):
        try:
            name = getattr(member, "name", None)
            if isinstance(name, Path):
                member.name = str(name)
        except Exception:
            pass
        return _orig_get_filtered_attrs(member, targetpath, tarinfo_isdir)

    try:
        tarfile._get_filtered_attrs = _patched_get_filtered_attrs  # type: ignore[attr-defined]
    except Exception:
        pass


def make_env(
    *,
    render_mode: Optional[str],
    track: Optional[str],
    agent_spec: Optional["AgentSpec"],
    wrappers: Optional[List[Callable[[gym.Env], gym.Env]]] = None,
    vecnormalize_stats_path: Optional[str] = None,
    training: bool = True,
) -> gym.Env:
    """Create a SuperTuxKart gym environment with optional wrappers and VecNormalize stats.

    Params:
    - render_mode: `None` for training, `"human"` for play
    - track: e.g., "hacienda"; if None, let env default
    - agent_spec: `pystk2_gymnasium.AgentSpec(use_ai=False)` typically
    - wrappers: list of callables `fn(env) -> env` applied in order
    - vecnormalize_stats_path: if provided, load stats into a new VecNormalize wrapping
      the env. `training` controls VecNormalize mode.

    Returns a ready-to-use `gym.Env`.
    """
    ensure_windows_dlls()
    _patch_tarfile_pathlike()

    import gymnasium as gym
    from pystk2_gymnasium import AgentSpec as _AgentSpec

    env_id = "supertuxkart/flattened_continuous_actions-v0"

    kwargs = {}
    if track is not None:
        kwargs["track"] = track
    if agent_spec is not None:
        kwargs["agent"] = agent_spec
    else:
        kwargs["agent"] = _AgentSpec(use_ai=False)

    base_env = gym.make(env_id, render_mode=render_mode, **kwargs)

    # Apply wrappers
    if wrappers:
        for wfn in wrappers:
            try:
                base_env = wfn(base_env)
            except Exception:
                # Be defensive: skip failing wrapper
                pass

    # Optionally wrap with VecNormalize using provided stats
    if vecnormalize_stats_path:
        try:
            base_env = VecNormalize.load(vecnormalize_stats_path, base_env)
            base_env.training = bool(training)
        except Exception:
            # Fall back to raw env if stats not found
            pass

    return base_env


__all__ = [
    "ensure_windows_dlls",
    "make_env",
]
