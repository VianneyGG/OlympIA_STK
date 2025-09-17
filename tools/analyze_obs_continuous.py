"""
Analyze continuous observations for STK env to correlate with progress and actions.

This script steps with random or heuristic actions for a short rollout, records
obs["continuous"], info["progress"/distance], and basic actions (steer, accel),
and prints simple correlation diagnostics to help map indices.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def ensure_windows_dlls() -> None:
    if os.name != "nt":
        return
    env_bin = Path(sys.prefix) / "Library" / "bin"
    if env_bin.is_dir():
        try:
            os.add_dll_directory(str(env_bin))
        except Exception:
            pass
    for stk_dir in [
        Path(r"C:\\Program Files\\SuperTuxKart 1.4"),
        Path(r"C:\\Program Files\\SuperTuxKart"),
    ]:
        if stk_dir.is_dir():
            try:
                os.add_dll_directory(str(stk_dir))
            except Exception:
                pass


def extract_progress(info: dict) -> float:
    p = info.get("progress")
    if p is not None:
        try:
            return float(p)
        except Exception:
            pass
    d = info.get("distance")
    if d is not None:
        try:
            return float(d)
        except Exception:
            pass
    lp = info.get("lap_progress")
    if lp is not None:
        try:
            return float(lp)
        except Exception:
            pass
    return 0.0


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--track", type=str, default="hacienda")
    parsed = parser.parse_args(args=args)

    ensure_windows_dlls()

    import gymnasium as gym
    from pystk2_gymnasium import AgentSpec

    env = gym.make(
        "supertuxkart/flattened_continuous_actions-v0",
        render_mode=None,
        agent=AgentSpec(use_ai=False),
        track=parsed.track,
    )

    obs, info = env.reset(seed=0)
    X = []
    dprog = []
    actions = []
    prev_p = extract_progress(info)

    for _ in range(parsed.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        vec = obs.get("continuous") if isinstance(obs, dict) else None
        if vec is None:
            continue
        vec = np.asarray(vec, dtype=float).reshape(-1)
        X.append(vec)
        p = extract_progress(info)
        dprog.append(p - prev_p)
        prev_p = p
        actions.append(np.asarray(action, dtype=float).reshape(-1))
        if terminated or truncated:
            obs, info = env.reset()
            prev_p = extract_progress(info)

    env.close()

    if not X:
        print("No continuous observations recorded.")
        return

    X = np.vstack(X)
    dprog = np.asarray(dprog)
    A = np.vstack(actions) if actions else None

    # Correlations with progress delta
    corrs = []
    for i in range(X.shape[1]):
        xi = X[:, i]
        try:
            c = float(np.corrcoef(xi, dprog)[0, 1])
        except Exception:
            c = 0.0
        corrs.append((i, c, float(np.std(xi))))
    corrs.sort(key=lambda t: abs(t[1]), reverse=True)

    print("Top 10 |corr| with d_progress:")
    for i, c, s in corrs[:10]:
        print(f"  idx={i:3d} corr={c:+.3f} std={s:.3g}")

    # Correlations with steer/accel if present
    if A is not None and A.shape[1] >= 2:
        steer = A[:, 0]
        accel = A[:, 1]
        sc = [
            (i, float(np.corrcoef(X[:, i], steer)[0, 1])) for i in range(X.shape[1])
        ]
        ac = [
            (i, float(np.corrcoef(X[:, i], accel)[0, 1])) for i in range(X.shape[1])
        ]
        sc.sort(key=lambda t: abs(t[1]), reverse=True)
        ac.sort(key=lambda t: abs(t[1]), reverse=True)
        print("Top 5 |corr| with steer:")
        for i, c in sc[:5]:
            print(f"  idx={i:3d} corr={c:+.3f}")
        print("Top 5 |corr| with accel:")
        for i, c in ac[:5]:
            print(f"  idx={i:3d} corr={c:+.3f}")


if __name__ == "__main__":
    main()
