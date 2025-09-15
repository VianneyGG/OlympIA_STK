"""
Single Player Training Scritraining_env = make_vec_env(
    "supertuxkart/flattened-v0", 
    n_envs=4,  # Réduit pour éviter les problèmes de mémoire
    env_kwargs={
        "render_mode": None,  # Pas de rendu pendant l'entraînement
        "track": "hacienda"   # Piste par défaut
    }
)perTuxKart
==============================================

Ce script entraîne un agent PPO (Proximal Policy Optimization) sur l'environnement 
SuperTuxKart de Gymnasium. Il utilise Stable Baselines3 pour l'implémentation.

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
# CONFIGURATION DE L'ENVIRONNEMENT
# =============================================================================

# Création d'un environnement vectorisé pour un entraînement parallèle
# SuperTuxKart : Environnement de course avec voitures
# n_envs=4 : Réduire le nombre d'environnements pour SuperTuxKart (plus lourd)
# Création d'un environnement vectorisé pour entraînement parallèle
# Utilise `make_vec_env` pour créer plusieurs copies de l'environnement afin
# d'améliorer l'efficacité de PPO. On garde le rendu désactivé pendant
# l'entraînement (render_mode=None) et on force la piste par défaut.
env_id = "supertuxkart/flattened_discrete-v0"
def _make_monitored_env():
    return Monitor(
        gym.make(env_id, render_mode=None, track="hacienda"),
        filename=None,
    )

# Nombre d'environnements parallèles (adaptez selon les ressources SSH)
n_envs = 8
# Crée le vec env, puis applique VecNormalize pour normaliser obs et récompenses
training_env = make_vec_env(_make_monitored_env, n_envs=n_envs)


# =============================================================================
# CONFIGURATION DU MODÈLE PPO
# =============================================================================

# Initialisation du modèle PPO avec une politique MultiInput (pour espaces d'observation dict)
# Policy network configuration for driving: separate heads for policy and value
policy_kwargs = dict(
    # SB3 >= 1.8: passer un dict directement (et non une liste contenant un dict)
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

## Evaluation Callback (SB3)
# Effectue une évaluation périodique pendant l'entraînement et enregistre le meilleur modèle et les logs d'évaluation

# Initialisation du modèle PPO avec une politique MultiInput (pour espaces d'observation dict)
model = PPO(
    policy="MultiInputPolicy",
    env=training_env,
    learning_rate=3e-4,
    n_steps=2048,        # timesteps per rollout (larger for stable updates)
    batch_size=64,
    n_epochs=10,         # SGD epochs per update
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=policy_kwargs,
    normalize_advantage=True,
    verbose=1,
    tensorboard_log="./tensorboard/ppo_stk/",
    device="auto",
)

# =============================================================================
# ENTRAÎNEMENT DU MODÈLE
# =============================================================================

# Définition du nom du modèle sauvegardé
model_name = "q-supertuxkart/flattened_discrete-v0-single_track_hacienda-ppos"
## Eval env for periodic evaluation during training
eval_env_cb = make_vec_env(_make_monitored_env, n_envs=1)

best_model_dir = os.path.join("ppo_stk", "best_model_discrete")
eval_log_dir = os.path.join("ppo_stk", "eval_logs_discrete")
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

model.learn(total_timesteps=int(2 * 1e6), progress_bar=True, callback=eval_callback)

# Export evaluation results to CSV for analysis
try:
    eval_npz_path = os.path.join(eval_log_dir, "evaluations.npz")
    if os.path.exists(eval_npz_path):
        data = np.load(eval_npz_path, allow_pickle=True)
        timesteps = data.get("timesteps", [])
        results = data.get("results", [])  # shape: (n_evals, n_eval_episodes)
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

# Sauvegarde du modèle et des statistiques de normalisation (VecNormalize)
model.save(model_name)
vn_stats_path = os.path.join("tensorboard", "ppo_stk", "vecnormalize.pkl")
try:
    os.makedirs(os.path.dirname(vn_stats_path), exist_ok=True)
    training_env.save(vn_stats_path)
except Exception:
    pass

# =============================================================================
# ÉVALUATION DU MODÈLE
# =============================================================================
# Recharge le modèle et réapplique les stats de normalisation pour évaluer en reward "réel"
model = PPO.load(model_name)

# Crée un env d'évaluation (1 env) et charge les stats de VecNormalize
eval_env = make_vec_env(_make_monitored_env, n_envs=1)
try:
    eval_env = VecNormalize.load(vn_stats_path, eval_env)
except Exception:
    # Si pas de fichier de stats (premier run), on garde l'env brut
    pass

# Désactive l'update des stats et n'applique pas la normalisation sur les rewards pour obtenir le score env réel
if isinstance(eval_env, VecNormalize):
    eval_env.training = False
    eval_env.norm_reward = False

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"Mean reward over 10 evaluation episodes: {mean_reward:.2f} +/- {std_reward:.2f}")