"""
Summarize q-supertuxkart/continuous/evaluations.npz contents with shapes and basic stats.

Usage (PowerShell):
  conda run --name OlympIA_STK_py311 python tools/summarize_evaluations.py
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def main() -> int:
    path = Path("q-supertuxkart/continuous/evaluations.npz")
    print(f"Loading: {path} exists={path.exists()}")
    if not path.exists():
        return 0

    data = np.load(path, allow_pickle=True)
    print("Keys:", list(data.files))

    def stats_array(a: np.ndarray) -> dict:
        try:
            if a.size and np.issubdtype(a.dtype, np.number):
                return {
                    "min": float(np.nanmin(a)),
                    "max": float(np.nanmax(a)),
                    "mean": float(np.nanmean(a)),
                    "std": float(np.nanstd(a)),
                }
        except Exception:
            pass
        return {}

    report = {}
    for k in data.files:
        a = data[k]
        rep = {
            "shape": getattr(a, "shape", None),
            "dtype": str(getattr(a, "dtype", "")),
            "stats": stats_array(a) if isinstance(a, np.ndarray) else {},
        }
        report[k] = rep

    # Try common keys for SB3 EvalCallback npz: 'timesteps', 'results' (N x M), 'ep_lengths'
    timesteps = data.get("timesteps") if "timesteps" in data.files else None
    if timesteps is not None:
        ts = np.asarray(timesteps).reshape(-1)
        report["__timesteps_preview__"] = ts[:10].tolist()

    results = data.get("results") if "results" in data.files else None
    if results is not None:
        R = np.asarray(results)
        # Heuristic: column 0 as reward if 2D; else flatten
        rewards = None
        if R.ndim == 2 and R.shape[1] >= 1:
            rewards = R[:, 0]
        elif R.ndim == 1:
            rewards = R
        if rewards is not None:
            r = np.asarray(rewards, dtype=float).reshape(-1)
            report["__rewards_summary__"] = {
                "count": int(r.size),
                "min": float(np.nanmin(r)) if r.size else None,
                "max": float(np.nanmax(r)) if r.size else None,
                "mean": float(np.nanmean(r)) if r.size else None,
                "std": float(np.nanstd(r)) if r.size else None,
                "last10_mean": float(np.nanmean(r[-10:])) if r.size else None,
            }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
