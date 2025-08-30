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

# =============================================================================
# IMPORTATIONS
# =============================================================================

import tarfile
from pathlib import Path
import os

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
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

# =============================================================================
# CONFIGURATION DE L'ENVIRONNEMENT
# =============================================================================

# Création d'un environnement vectorisé pour un entraînement parallèle
# SuperTuxKart : Environnement de course avec voitures
# n_envs=4 : Réduire le nombre d'environnements pour SuperTuxKart (plus lourd)
training_env = DummyVecEnv([lambda: Monitor(
    gym.make(
        "supertuxkart/flattened_discrete-v0", 
        render_mode=None,
        track="hacienda"
    ),
    filename=None
)])

# =============================================================================
# CONFIGURATION DU MODÈLE PPO
# =============================================================================

# Initialisation du modèle PPO avec une politique MultiInput (pour espaces d'observation dict)
model = PPO(
    policy="MultiInputPolicy",             # Type de politique : réseau de neurones pour espaces d'observation dict
    env=training_env,                      # Environnement d'entraînement
    verbose=1                              # Niveau de verbosité (1 = progression de l'entraînement)
)

# =============================================================================
# ENTRAÎNEMENT DU MODÈLE
# =============================================================================

# Définition du nom du modèle sauvegardé
model_name = "ppo_stk"

model.learn(total_timesteps=int(1e6))
model.save(model_name)

# =============================================================================
# ÉVALUATION DU MODÈLE
# =============================================================================

model = PPO.load(model_name)