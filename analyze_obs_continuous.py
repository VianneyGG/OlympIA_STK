import os
import sys
from pathlib import Path
import numpy as np
import gymnasium as gym

# Ensure DLLs on Windows
env_bin = Path(sys.prefix) / "Library" / "bin"
if env_bin.is_dir():
    try:
        os.add_dll_directory(str(env_bin))
    except Exception:
        pass
for stk_dir in [
    Path(r"C:\Program Files\SuperTuxKart 1.4"),
    Path(r"C:\Program Files\SuperTuxKart"),
]:
    if stk_dir.is_dir():
        try:
            os.add_dll_directory(str(stk_dir))
        except Exception:
            pass

from pystk2_gymnasium import AgentSpec


def run_probe(steps: int = 1500, seed: int = 0):
    env_id = "supertuxkart/flattened_continuous_actions-v0"
    env = gym.make(env_id, render_mode=None, agent=AgentSpec(use_ai=False), track="hacienda", num_kart=2)
    obs, info = env.reset(seed=seed)
    cont_dim = None
    xs = []
    dists = []
    actions = []
    if isinstance(obs, dict) and "continuous" in obs:
        cont = obs["continuous"]
        cont_dim = cont.shape[-1]
    for t in range(steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        if isinstance(obs, dict) and "continuous" in obs:
            xs.append(np.asarray(obs["continuous"], dtype=float))
        dists.append(float(info.get("distance", 0.0)))
        actions.append(np.asarray(a, dtype=float))
        if term or trunc:
            break
    env.close()

    X = np.vstack(xs) if xs else None
    A = np.vstack(actions) if actions else None
    D = np.asarray(dists, dtype=float)
    dD = np.concatenate([[0.0], np.diff(D)])

    print("continuous dim:", cont_dim)
    if X is None:
        print("No continuous obs captured.")
        return

    # Correlation of each index with distance delta
    corrs = []
    for i in range(X.shape[1]):
        xi = X[:, i]
        if np.std(xi) < 1e-8:
            c = 0.0
        else:
            c = float(np.corrcoef(xi, dD)[0, 1])
        corrs.append((i, c, float(np.std(xi))))
    corrs_sorted = sorted(corrs, key=lambda t: abs(t[1]), reverse=True)
    print("Top 10 indices by |corr(x_i, dDistance)| (index, corr, std):")
    for i, c, s in corrs_sorted[:10]:
        print(f"  {i:3d}  corr={c:+.4f}  std={s:.4e}")

    # Look for yaw sin/cos pair: mean(x^2 + y^2) ~ 1
    n = X.shape[1]
    best_pair = None
    best_err = 1e9
    for i in range(n):
        xi2 = X[:, i] ** 2
        for j in range(i + 1, n):
            s = xi2 + X[:, j] ** 2
            err = float(abs(np.mean(s) - 1.0))
            if err < best_err:
                best_err = err
                best_pair = (i, j, np.mean(s), np.std(s))
    if best_pair:
        i, j, mean_s, std_s = best_pair
        print(f"Best sin/cos-like pair: (i={i}, j={j}) with mean(x_i^2+x_j^2)={mean_s:.3f}, std={std_s:.3e}")

    # Correlation with actions to guess steering/throttle axes
    if A is not None and A.shape[1] >= 2:
        steer = A[:, 0]
        accel = A[:, 1]
        # Which indices move with steering?
        steer_corr = []
        accel_corr = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            c_s = 0.0 if np.std(xi) < 1e-8 else float(np.corrcoef(xi, steer)[0, 1])
            c_a = 0.0 if np.std(xi) < 1e-8 else float(np.corrcoef(xi, accel)[0, 1])
            steer_corr.append((i, c_s))
            accel_corr.append((i, c_a))
        print("Top 5 indices by |corr(x_i, steer)|:")
        for i, c in sorted(steer_corr, key=lambda t: abs(t[1]), reverse=True)[:5]:
            print(f"  {i:3d}  corr={c:+.4f}")
        print("Top 5 indices by |corr(x_i, accel)|:")
        for i, c in sorted(accel_corr, key=lambda t: abs(t[1]), reverse=True)[:5]:
            print(f"  {i:3d}  corr={c:+.4f}")

    # Heuristic suggestion for forward_speed index: highest positive corr with dD
    cand = next((i for i, c, s in corrs_sorted if c > 0.2 and s > 1e-6), None)
    if cand is not None:
        print("Suggested forward_speed index (heuristic):", cand)

    # Heuristic suggestion for off_track: near-binary feature
    # Compute a binariness score and try to correlate with (lack of) progress
    dp = dD
    prog_flag = (dp > 1e-4).astype(float)
    best_score = -1e9
    best_off = None
    for i in range(X.shape[1]):
        xi = X[:, i]
        # Scale to [0,1] and check how often it's near 0 or 1
        xmin, xmax = float(np.min(xi)), float(np.max(xi))
        rng = xmax - xmin
        if rng < 1e-9:
            continue
        xn = (xi - xmin) / rng
        bin_score = float(np.mean((xn < 0.05) | (xn > 0.95)))
        # Correlate with NOT progressing (rough proxy for off-track)
        try:
            c_stop = float(np.corrcoef((xn > 0.5).astype(float), (1.0 - prog_flag))[0, 1])
        except Exception:
            c_stop = 0.0
        score = bin_score + abs(c_stop)
        if score > best_score:
            best_score = score
            best_off = (i, bin_score, c_stop)
    if best_off is not None:
        i, b, c = best_off
        print(f"Suggested off_track index (heuristic): {i}  binariness~{b:.3f}  corr_stop~{c:+.3f}")

    # Heuristic suggestion for lateral_speed: high variance, low |corr| with dD
    lat_candidates = [(i, abs(c), s) for i, c, s in corrs]
    lat_candidates.sort(key=lambda t: (t[1], -t[2]))  # prioritize small |corr|, then larger std
    if lat_candidates:
        best_lat = next(((i, c, s) for i, c, s in lat_candidates if c < 0.15 and s > 1e-3), lat_candidates[0])
        i, c, s = best_lat
        print(f"Suggested lateral_speed index (heuristic): {i}  |corr|={c:.3f}  std={s:.3e}")

if __name__ == "__main__":
    run_probe()
