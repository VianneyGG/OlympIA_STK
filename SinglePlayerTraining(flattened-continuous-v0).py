"""
Single Player Training – SuperTuxKart (continuous)
=================================================

Ce script entraîne un agent PPO (Proximal Policy Optimization) sur l'environnement
SuperTuxKart via Gymnasium, avec actions continues. Il utilise Stable Baselines3.

Objectifs de stabilité/rendement ajoutés:
- Normalisation des observations et des récompenses (VecNormalize)
- Exploration renfoncée (use_sde + ent_coef plus élevé)
- Clipping de la value (clip_range_vf) et target_kl pour éviter les dérives

Auteur: [Votre nom]
Date: Août 2025
"""


# =============================
# PATCH: Ajout DLL directories
# =============================
import os
import sys
from pathlib import Path
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

# =============================================================================
# IMPORTATIONS
# =============================================================================

import tarfile
from pathlib import Path
import os
from typing import Optional, List, Dict, Any
from collections import Counter
import csv

# Compatibility patch: some environments (or third-party packages) may pass
# pathlib.Path / PosixPath objects as tar member names to the stdlib
# `tarfile` internals. In Python 3.9 the internal function
# `_get_filtered_attrs` expects `member.name` to be a string and calls
# `startswith` on it. When it receives a Path, it raises
# AttributeError: 'PosixPath' object has no attribute 'startswith'.
#
# We monkeypatch `tarfile._get_filtered_attrs` to coerce `member.name` to
# `str` when necessary. This is a minimal, local change that avoids
# editing installed packages and fixes the crash seen over SSH when
# `pystk2_gymnasium` extracts tracks or archives.
_orig_get_filtered_attrs = getattr(tarfile, "_get_filtered_attrs", None)
if _orig_get_filtered_attrs is not None:
    def _patched_get_filtered_attrs(member, targetpath, tarinfo_isdir=False):
        try:
            name = getattr(member, "name", None)
            if isinstance(name, Path):
                member.name = str(name)
        except Exception:
            # Be defensive: if anything goes wrong, fall back to original
            pass
        return _orig_get_filtered_attrs(member, targetpath, tarinfo_isdir)

    tarfile._get_filtered_attrs = _patched_get_filtered_attrs

import gymnasium as gym
import pystk2_gymnasium as pystk_gym
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

# =============================================================================

# =============================================================================
# LOGIQUE PRINCIPALE (multiprocessing safe)
# =============================================================================

if __name__ == "__main__":
    # CONFIGURATION DE L'ENVIRONNEMENT
    env_id = "supertuxkart/flattened_continuous_actions-v0"
    def _make_monitored_env():
        return Monitor(
            gym.make(env_id, render_mode=None, track="hacienda", num_kart=2),
            filename=None,
        )

    n_envs = 8
    training_env = make_vec_env(_make_monitored_env, n_envs=n_envs)
    training_env = VecNormalize(
        training_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        norm_obs_keys=["continuous"],
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=training_env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        normalize_advantage=True,
        use_sde=True,
        sde_sample_freq=64,
        clip_range_vf=1.0,
        target_kl=0.02,
        verbose=1,
        tensorboard_log="./tensorboard/ppo_stk_continuous/",
        device="auto",
    )

    model_name = "q-supertuxkart/flattened_continuous-v0-single_track_hacienda-ppos"
    eval_env_cb = make_vec_env(_make_monitored_env, n_envs=1)
    eval_env_cb = VecNormalize(eval_env_cb, training=False, norm_obs=True, norm_reward=False, norm_obs_keys=["continuous"])

    best_model_dir = os.path.join("ppo_stk", "best_model_continuous")
    eval_log_dir = os.path.join("ppo_stk", "eval_logs_continuous")
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(eval_log_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env=eval_env_cb,
        best_model_save_path=best_model_dir,
        log_path=eval_log_dir,
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=int(2 * 1e6),
        progress_bar=True,
        callback=eval_callback,
        tb_log_name="ppo_stk_continuous",
    )

    try:
        eval_npz_path = os.path.join(eval_log_dir, "evaluations.npz")
        if os.path.exists(eval_npz_path):
            data = np.load(eval_npz_path, allow_pickle=True)
            timesteps = data.get("timesteps", [])
            results = data.get("results", [])
            out_csv = os.path.join(eval_log_dir, "evaluations.csv")
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timesteps", "mean_reward", "std_reward"])
                if len(timesteps) == len(results):
                    for t, rs in zip(timesteps, results):
                        rs = np.asarray(rs, dtype=float).reshape(-1)
                        w.writerow([int(t), float(np.mean(rs)), float(np.std(rs))])
    except Exception:
        pass

    model.save(model_name)
    vn_stats_path = os.path.join("tensorboard", "ppo_stk", "vecnormalize_cont.pkl")
    try:
        os.makedirs(os.path.dirname(vn_stats_path), exist_ok=True)
        training_env.save(vn_stats_path)
    except Exception:
        pass

    # ÉVALUATION DU MODÈLE
    model = PPO.load(model_name)
    eval_env = make_vec_env(_make_monitored_env, n_envs=1)
    try:
        eval_env = VecNormalize.load(vn_stats_path, eval_env)
    except Exception:
        pass
    if isinstance(eval_env, VecNormalize):
        eval_env.training = False
        eval_env.norm_reward = False
        eval_env.norm_obs_keys = ["continuous"]
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward over 10 evaluation episodes: {mean_reward:.2f} +/- {std_reward:.2f}")