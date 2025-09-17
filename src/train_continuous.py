from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure repo root is importable so `utils/...` can be found when running this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env_utils import ensure_windows_dlls, make_env
from callbacks import RewardComponentsLogger, make_checkpoint_callback, make_eval_callback


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    # Env
    env_id: str = "supertuxkart/flattened_continuous_actions-v0"
    track: str = "hacienda"
    n_envs: int = 4
    seed: int = 42
    # Paths
    tb_log_dir: str = "tensorboard/ppo_stk_continuous"
    run_name: str = "ppo_stk_continuous_1"
    out_dir: str = "q-supertuxkart/continuous"
    checkpoints_dir: Optional[str] = "q-supertuxkart/continuous/checkpoints"
    # VecNormalize
    norm_obs_keys: Optional[list] = None  # e.g., ["continuous"]
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    gamma: float = 0.99
    # PPO hyperparams
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_range_vf: float = 1.0
    target_kl: float = 0.02
    use_sde: bool = True
    sde_sample_freq: int = 64
    total_timesteps: int = int(7e6)
    # UI
    progress_bar: bool = True
    # Callbacks
    eval_freq: int = 10_000
    n_eval_episodes: int = 5
    checkpoint_freq: int = 0
    # Reward shaping
    enable_reward_shaping: bool = True
    # Optional mapping for STKRewardShaping from observations indices
    obs_index_map: Optional[Dict[str, int]] = None


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Provide defaults
    cfg = TrainConfig()
    for k, v in (data or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def build_wrappers(cfg: TrainConfig):
    from stable_baselines3.common.monitor import Monitor
    wrappers = []
    if cfg.enable_reward_shaping:
        # Import lazily to avoid dependency at module import
        from utils.reward_wrappers import STKRewardShaping

        def _shape(env):
            return STKRewardShaping(env, obs_index_map=cfg.obs_index_map)

        wrappers.append(_shape)

    def _monitor(env):
        return Monitor(env)

    wrappers.append(_monitor)
    return wrappers


def main(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train PPO on STK continuous actions")
    parser.add_argument("--config", type=str, default="configs/ppo_continuous.yaml")
    parsed = parser.parse_args(args=args)

    ensure_windows_dlls()
    # Prefer plain tqdm over Rich to avoid rare shutdown ImportError from rich progress
    os.environ.setdefault("TQDM_DISABLE_RICH", "1")
    # Headless-friendly defaults (Linux CI/servers): no DISPLAY => use SDL dummy drivers
    if os.name != "nt":
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    cfg = load_config(parsed.config)
    set_global_seeds(int(cfg.seed))

    # Paths
    tb_dir = Path(cfg.tb_log_dir) / cfg.run_name
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path(cfg.checkpoints_dir) if cfg.checkpoints_dir else Path())
    if cfg.checkpoints_dir:
        Path(cfg.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # Build vectorized training env
    wrappers = build_wrappers(cfg)

    def _make_train_env():
        from pystk2_gymnasium import AgentSpec

        return make_env(
            render_mode=None,
            track=cfg.track,
            agent_spec=AgentSpec(use_ai=False),
            wrappers=wrappers,
        )

    vec = make_vec_env(_make_train_env, n_envs=int(cfg.n_envs), seed=int(cfg.seed))

    vec = VecNormalize(
        vec,
        norm_obs=True,
        norm_reward=True,
        gamma=float(cfg.gamma),
        clip_obs=float(cfg.clip_obs),
        clip_reward=float(cfg.clip_reward),
        norm_obs_keys=(cfg.norm_obs_keys or ["continuous"]),
    )

    # Policy architecture
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        policy="MultiInputPolicy",
        env=vec,
        learning_rate=float(cfg.learning_rate),
        n_steps=int(cfg.n_steps),
        batch_size=int(cfg.batch_size),
        n_epochs=int(cfg.n_epochs),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_range=float(cfg.clip_range),
        ent_coef=float(cfg.ent_coef),
        vf_coef=float(cfg.vf_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        policy_kwargs=policy_kwargs,
        normalize_advantage=True,
        use_sde=bool(cfg.use_sde),
        sde_sample_freq=int(cfg.sde_sample_freq),
        clip_range_vf=float(cfg.clip_range_vf),
        target_kl=float(cfg.target_kl),
        verbose=1,
        tensorboard_log=str(tb_dir.parent),
        device="auto",
        seed=int(cfg.seed),
    )

    # Eval env shares VecNormalize stats via `load` in a fresh env in eval mode
    def _make_eval_env():
        from pystk2_gymnasium import AgentSpec

        return make_env(
            render_mode=None,
            track=cfg.track,
            agent_spec=AgentSpec(use_ai=False),
            wrappers=build_wrappers(cfg),
        )

    # Match eval env type with training env (VecNormalize) and copy running statistics
    # First, vectorize eval env (DummyVecEnv with 1 env)
    eval_vec = make_vec_env(_make_eval_env, n_envs=1, seed=int(cfg.seed))
    eval_env = VecNormalize(
        eval_vec,
        norm_obs=True,
        norm_reward=False,  # don't normalize rewards for evaluation metrics
        gamma=float(cfg.gamma),
        clip_obs=float(cfg.clip_obs),
    )
    # Share observation/return RMS from the training VecNormalize wrapper
    try:
        eval_env.obs_rms = vec.obs_rms
        eval_env.ret_rms = vec.ret_rms
        eval_env.training = False
    except Exception:
        pass

    # Callbacks
    eval_cb = make_eval_callback(
        eval_env=eval_env,
        best_model_save_path=str(out_dir),
        log_path=str(out_dir),
        eval_freq=int(cfg.eval_freq),
        n_eval_episodes=int(cfg.n_eval_episodes),
        deterministic=True,
        render=False,
    )

    ckpt_cb = make_checkpoint_callback(cfg.checkpoint_freq, save_path=str(Path(cfg.checkpoints_dir) if cfg.checkpoints_dir else out_dir))

    callbacks = [eval_cb, RewardComponentsLogger()]
    if ckpt_cb is not None:
        callbacks.append(ckpt_cb)
    cb_list = CallbackList(callbacks)

    # Train
    model.learn(
        total_timesteps=int(cfg.total_timesteps),
        progress_bar=bool(cfg.progress_bar),
        callback=cb_list,
        tb_log_name=str(tb_dir.name),
    )

    # Determine TensorBoard run number (directory suffix like '<run_name>_3')
    def _tb_run_number() -> str:
        base = tb_dir.parent
        name = tb_dir.name
        try:
            candidates = [d for d in base.iterdir() if d.is_dir() and d.name.startswith(name)]
            if not candidates:
                return ""
            # pick most recently modified
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            chosen = candidates[0].name
            if chosen == name:
                return ""
            if chosen.startswith(name + "_"):
                return chosen[len(name) + 1 :]
            return ""
        except Exception:
            return ""

    run_no = _tb_run_number()
    step_no = int(getattr(model, "num_timesteps", 0))

    # Save model and VecNormalize stats with TB run number and step
    suffix = (f"_{run_no}" if run_no else "")
    best_model_path = out_dir / f"best_model{suffix}.zip"
    latest_model_path = out_dir / f"latest_model{suffix}_{step_no}.zip"
    model.save(str(latest_model_path))

    vec_stats_path = out_dir / f"vecnormalize{suffix}_{step_no}.pkl"
    try:
        vec.save(str(vec_stats_path))
    except Exception:
        pass

    print(f"Saved latest model to: {latest_model_path}")
    print(f"VecNormalize stats: {vec_stats_path if vec_stats_path.exists() else 'not saved'}")


if __name__ == "__main__":
    main()
