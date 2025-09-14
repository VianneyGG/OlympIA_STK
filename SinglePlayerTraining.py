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
from typing import Optional, List, Dict, Any

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
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

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

# =============================================================================
# CALLBACK POUR LE TRACE DE LA LOSS
# =============================================================================

class LossPlotCallback(BaseCallback):
    """
    Callback SB3 minimaliste pour enregistrer et tracer `train/loss` pendant l'entraînement
    (cf. docs SB3: train/loss est loggé pour PPO). Sauvegarde un PNG en fin d'entraînement.
    """

    def __init__(
        self,
        save_path: Optional[str] = os.path.join("tensorboard", "ppo_stk", "loss_plot.png"),
    ) -> None:
        super().__init__()
        self.save_path = save_path
        self.timesteps: List[int] = []
        self.loss_values: List[float] = []

    def _get_logger_values(self) -> Dict[str, float]:
        """Récupère les dernières valeurs du logger SB3 (impl-dépendant)."""
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return {}
        for attr in ("name_to_value", "_name_to_value"):
            values = getattr(logger, attr, None)
            if isinstance(values, dict) and values:
                return {str(k): v for k, v in values.items() if isinstance(k, str)}
        get_log_dict = getattr(logger, "get_log_dict", None)
        if callable(get_log_dict):
            try:
                values = get_log_dict()
                if isinstance(values, dict):
                    return values
            except Exception:
                pass
        return {}

    def _on_training_start(self) -> None:
        self.timesteps.clear()
        self.loss_values.clear()

    def _on_rollout_start(self) -> None:
        # Les valeurs `train/...` du logger correspondent à l'update précédent.
        values = self._get_logger_values()
        if not values:
            return
        # Priorité aux clés documentées: `train/loss`, fallback `train/value_loss`.
        val = values.get("train/loss")
        if val is None:
            val = values.get("train/value_loss")
        if val is None:
            return
        try:
            self.timesteps.append(int(self.model.num_timesteps))
            self.loss_values.append(float(val))
        except Exception:
            pass

    def _on_training_end(self) -> None:
        if not self.timesteps or not self.loss_values:
            return
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(self.timesteps, self.loss_values, label="train/loss")
            ax.set_title("Training Loss")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Loss")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(self.save_path)
            plt.close(fig)
        except Exception:
            pass

    def _on_step(self) -> bool:
        # Requis par BaseCallback; ici, rien à faire à chaque step.
        return True


def loss_callback() -> LossPlotCallback:
    """Fonction utilitaire pour créer une instance de callback de loss.

    Retourne:
        LossPlotCallback: callback compatible avec `learn(callback=...)`.
    """
    return LossPlotCallback()

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
    ent_coef=0.01,
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

model.learn(total_timesteps=int(1e6), progress_bar=True, callback=loss_callback())
model.save(model_name)

# =============================================================================
# ÉVALUATION DU MODÈLE
# =============================================================================

model = PPO.load(model_name)