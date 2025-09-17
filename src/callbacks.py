from __future__ import annotations

from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback


class RewardComponentsLogger(BaseCallback):
    """Log average reward components from info["rew_components"] to TensorBoard.

    Works both with vectorized and non-vectorized envs; it inspects `self.locals["infos"]`
    which is present during rollout collection in SB3.
    """

    def __init__(self, verbose: int = 0, prefix: str = "rewards/"):
        super().__init__(verbose)
        self.prefix = prefix.rstrip("/") + "/"

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not isinstance(infos, (list, tuple)) or not infos:
            return True

        sums = {}
        counts = {}
        for info in infos:
            if not isinstance(info, dict):
                continue
            comps = info.get("rew_components")
            if not isinstance(comps, dict):
                continue
            for k, v in comps.items():
                try:
                    fv = float(v)
                except Exception:
                    continue
                sums[k] = sums.get(k, 0.0) + fv
                counts[k] = counts.get(k, 0) + 1

        for k, s in sums.items():
            c = max(1, counts.get(k, 1))
            mean_v = s / c
            self.logger.record(self.prefix + k, mean_v)
        return True


def make_eval_callback(
    eval_env,
    best_model_save_path: str,
    log_path: str,
    eval_freq: int,
    n_eval_episodes: int = 5,
    deterministic: bool = True,
    render: bool = False,
) -> EvalCallback:
    """Create a preconfigured EvalCallback with sensible defaults."""
    return EvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_save_path,
        log_path=log_path,
        eval_freq=int(eval_freq),
        n_eval_episodes=int(n_eval_episodes),
        deterministic=deterministic,
        render=render,
    )


def make_checkpoint_callback(
    save_freq: Optional[int],
    save_path: str,
    name_prefix: str = "rl_model",
) -> Optional[CheckpointCallback]:
    """Optionally create a checkpoint callback if `save_freq` is set (>0)."""
    if not save_freq or save_freq <= 0:
        return None
    return CheckpointCallback(save_freq=int(save_freq), save_path=save_path, name_prefix=name_prefix)


__all__ = [
    "RewardComponentsLogger",
    "make_eval_callback",
    "make_checkpoint_callback",
]
