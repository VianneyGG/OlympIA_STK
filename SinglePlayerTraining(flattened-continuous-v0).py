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
env_id = "supertuxkart/flattened_continuous_actions-v0"
def _make_monitored_env():
    return Monitor(
        gym.make(env_id, render_mode=None, track="hacienda", num_kart=1),
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

# =============================================================================
# CALLBACK POUR LE TRACE DE LA LOSS
# =============================================================================

class TrainingMetricsCallback(BaseCallback):
    """
    Callback d'instrumentation avancée pour PPO:
    - Récupère toutes les métriques `logger` SB3 clés à chaque update.
    - Agrège des stats d'épisodes (récompenses, longueurs) via `infos`.
    - Suit la distribution des actions (en Discrete) pour détecter un effondrement de politique.
    - Sauvegarde un CSV, des graphiques (loss + overview), et un rapport de diagnostics.

    Référence logger SB3: https://stable-baselines3.readthedocs.io/en/master/common/logger.html
    """

    def __init__(
        self,
        out_dir: Optional[str] = os.path.join("tensorboard", "ppo_stk"),
        loss_plot_filename: str = "loss_plot.png",
        overview_plot_filename: str = "metrics_overview.png",
        csv_filename: str = "training_metrics.csv",
        diag_filename: str = "training_diagnostics.txt",
    ) -> None:
        super().__init__()
        self.out_dir = out_dir
        self.loss_plot_path = os.path.join(out_dir, loss_plot_filename)
        self.overview_plot_path = os.path.join(out_dir, overview_plot_filename)
        self.csv_path = os.path.join(out_dir, csv_filename)
        self.diag_path = os.path.join(out_dir, diag_filename)

        # time series storage per update
        self.timesteps: List[int] = []
        self.metric_rows: List[Dict[str, Any]] = []
        self.loss_values: List[float] = []

        # episode stats and action stats (aggregated across training)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.action_counter: Counter[int] = Counter()
        self.total_actions: int = 0

    # ---------- Helpers ----------
    def _get_logger_values(self) -> Dict[str, float]:
        """Récupère les dernières valeurs du logger SB3 (de façon robuste)."""
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

    def _record_update_metrics(self) -> None:
        """Prend un snapshot des métriques logger et pousse dans la série temporelle."""
        values = self._get_logger_values()
        if not values:
            return
        step = int(getattr(self.model, "num_timesteps", 0))
        row: Dict[str, Any] = {"timesteps": step}
        # On garde toutes les clés disponibles (train/..., rollout/..., time/...).
        for k, v in values.items():
            try:
                row[k] = float(v)
            except Exception:
                # garde brut si non castable
                row[k] = v
        self.timesteps.append(step)
        self.metric_rows.append(row)

        # extrait et stocke la loss si dispo (pour le graphe dédié)
        loss_val = values.get("train/loss")
        if loss_val is None:
            loss_val = values.get("train/value_loss")
        if loss_val is not None:
            try:
                self.loss_values.append(float(loss_val))
            except Exception:
                pass

    # ---------- Hooks ----------
    def _on_training_start(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        self.timesteps.clear()
        self.metric_rows.clear()
        self.loss_values.clear()
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.action_counter.clear()
        self.total_actions = 0

    def _on_rollout_start(self) -> None:
        # Les valeurs logger à ce moment reflètent l'update précédent
        self._record_update_metrics()

    def _on_step(self) -> bool:
        # Récupération prudente des infos et actions pendant la collecte de rollouts
        try:
            infos = None
            actions = None
            if isinstance(self.locals, dict):
                infos = self.locals.get("infos", None)
                actions = self.locals.get("actions", None)

            # Agrège stats d'épisodes depuis Monitor (infos[*]['episode'])
            if infos is not None:
                # infos est souvent une liste/tuple de taille n_envs
                for info in (infos if isinstance(infos, (list, tuple)) else [infos]):
                    if not isinstance(info, dict):
                        continue
                    ep = info.get("episode") or info.get("ep_info")
                    if isinstance(ep, dict):
                        r = ep.get("r")
                        l = ep.get("l")
                        if r is not None:
                            try:
                                self.episode_rewards.append(float(r))
                            except Exception:
                                pass
                        if l is not None:
                            try:
                                self.episode_lengths.append(int(l))
                            except Exception:
                                pass

            # Comptage des actions (Discrete)
            if actions is not None:
                try:
                    arr = np.asarray(actions).reshape(-1)
                    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                        for a in arr.tolist():
                            self.action_counter[int(a)] += 1
                            self.total_actions += 1
                except Exception:
                    pass
        except Exception:
            # Ne jamais interrompre l'entraînement à cause du callback
            return True
        return True

    def _on_rollout_end(self) -> None:
        # Certaines implémentations mettent à jour des métriques ici aussi
        self._record_update_metrics()

    def _write_csv(self) -> None:
        if not self.metric_rows:
            return
        # Collecter l'ensemble des colonnes
        all_keys: List[str] = ["timesteps"]
        seen = set(all_keys)
        for row in self.metric_rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                for row in self.metric_rows:
                    writer.writerow({k: row.get(k, "") for k in all_keys})
        except Exception:
            pass

    def _plot_loss(self) -> None:
        if not self.timesteps or not self.loss_values:
            return
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            # Aligner approximativement en X
            xs = self.timesteps[: len(self.loss_values)]
            ax.plot(xs, self.loss_values, label="train/loss")
            ax.set_title("Training Loss")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Loss")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(self.loss_plot_path)
            plt.close(fig)
        except Exception:
            pass

    def _plot_overview(self) -> None:
        if not self.metric_rows:
            return
        try:
            # Extraire séries courantes
            key_groups = [
                ["train/loss", "train/value_loss"],
                ["rollout/ep_rew_mean", "rollout/ep_len_mean"],
                ["train/approx_kl", "train/clip_fraction"],
                ["train/explained_variance"],
            ]
            steps = [row.get("timesteps", 0) for row in self.metric_rows]
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
            axes = axes.ravel()
            for ax, keys in zip(axes, key_groups):
                plotted = False
                for k in keys:
                    ys = [row.get(k, None) for row in self.metric_rows]
                    ys = [float(y) if isinstance(y, (int, float)) else np.nan for y in ys]
                    if np.isfinite(ys).any():
                        ax.plot(steps, ys, label=k)
                        plotted = True
                if plotted:
                    ax.set_title(", ".join(keys))
                    ax.set_xlabel("Timesteps")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(self.overview_plot_path)
            plt.close(fig)
        except Exception:
            pass

    def _write_diagnostics(self) -> None:
        try:
            lines: List[str] = []
            lines.append("Diagnostics d'entraînement PPO\n")
            lines.append(f"Total timesteps: {getattr(self.model, 'num_timesteps', 0)}\n")

            # Récupération de quelques métriques terminales (dernière ligne)
            last = self.metric_rows[-1] if self.metric_rows else {}
            def fmt(k, default="n/a"):
                v = last.get(k, None)
                return f"{v:.6f}" if isinstance(v, (int, float)) else str(v or default)

            lines.append("Dernières métriques:\n")
            for k in [
                "rollout/ep_rew_mean",
                "rollout/ep_len_mean",
                "train/loss",
                "train/value_loss",
                "train/entropy_loss",
                "train/policy_gradient_loss",
                "train/approx_kl",
                "train/clip_fraction",
                "train/explained_variance",
                "time/fps",
            ]:
                lines.append(f"  - {k}: {fmt(k)}\n")

            # Heuristiques simples
            def moving_avg(x: List[float], n: int) -> List[float]:
                if n <= 1 or len(x) < n:
                    return x[:]
                ret = []
                s = 0.0
                for i, v in enumerate(x):
                    s += v
                    if i >= n:
                        s -= x[i - n]
                    if i >= n - 1:
                        ret.append(s / n)
                return ret

            ep_rew = [row.get("rollout/ep_rew_mean") for row in self.metric_rows if isinstance(row.get("rollout/ep_rew_mean"), (int, float))]
            ev = [row.get("train/explained_variance") for row in self.metric_rows if isinstance(row.get("train/explained_variance"), (int, float))]
            kl = [row.get("train/approx_kl") for row in self.metric_rows if isinstance(row.get("train/approx_kl"), (int, float))]
            clip_frac = [row.get("train/clip_fraction") for row in self.metric_rows if isinstance(row.get("train/clip_fraction"), (int, float))]

            lines.append("\nObservations:\n")
            # Plateau de récompense
            if len(ep_rew) >= 30:
                ma = moving_avg(ep_rew, 10)
                first = np.nanmean(ma[:3]) if len(ma) >= 3 else np.nan
                last_ma = np.nanmean(ma[-3:]) if len(ma) >= 3 else np.nan
                delta = last_ma - first
                lines.append(f"- Variation récompense (moyenne mobile 10): {delta:.3f}\n")
                if abs(delta) < 0.05 * (abs(first) + 1e-6):
                    lines.append("  -> Récompense quasi stable: possible plateau / learning_rate trop faible / exploration insuffisante.\n")

            # Explained variance faible
            if ev:
                ev_last = ev[-1]
                if ev_last is not None and ev_last < 0.2:
                    lines.append("- Explained variance faible (<0.2): valeur prédictive du critic faible (sous-apprentissage du critic).\n")

            # KL et clip fraction
            if kl:
                kl_last = kl[-1]
                if kl_last is not None and (kl_last < 1e-4 or kl_last > 0.2):
                    lines.append("- approx_kl extrême: politique potentiellement figée (trop bas) ou updates trop agressifs (trop haut). Ajuster clip_range/ent_coef/lr.\n")
            if clip_frac:
                cf_last = clip_frac[-1]
                if cf_last is not None and (cf_last < 0.01 or cf_last > 0.5):
                    lines.append("- clip_fraction extrême: gradients peut-être annulés (trop bas) ou clipping fréquent (trop haut).\n")

            # Distribution des actions
            if self.total_actions > 0 and len(self.action_counter) > 0:
                most_common = self.action_counter.most_common(1)[0]
                top1_frac = most_common[1] / max(1, self.total_actions)
                lines.append(f"- Action la plus fréquente: {most_common[0]} ({top1_frac*100:.1f}% des actions)\n")
                if top1_frac > 0.9:
                    lines.append("  -> Effondrement de politique probable (peu d'exploration). Augmenter ent_coef ou vérifier le reward shaping.\n")

            with open(self.diag_path, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception:
            pass

    def _plot_action_hist(self) -> None:
        if self.total_actions <= 0 or not self.action_counter:
            return
        try:
            actions, counts = zip(*sorted(self.action_counter.items()))
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(actions, counts)
            ax.set_title("Distribution des actions (entraînement)")
            ax.set_xlabel("Action (Discrete)")
            ax.set_ylabel("Occurrences")
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "action_counts.png"))
            plt.close(fig)
        except Exception:
            pass

    def _on_training_end(self) -> None:
        # Snap final metrics once more (si dispo)
        self._record_update_metrics()
        # Persistance
        self._write_csv()
        self._plot_loss()  # garde compat avec chemin `loss_plot.png`
        self._plot_overview()
        self._plot_action_hist()
        self._write_diagnostics()


def loss_callback() -> TrainingMetricsCallback:
    """Instancie le callback d'instrumentation avancée.

    Retourne:
        TrainingMetricsCallback: callback compatible avec `learn(callback=...)`.
    """
    return TrainingMetricsCallback()

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
model_name = "q-supertuxkart/flattened_continuous-v0-single_track_hacienda-ppos"

model.learn(total_timesteps=int(2 * 1e6), progress_bar=True, callback=loss_callback())

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