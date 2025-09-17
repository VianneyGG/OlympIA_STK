from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from env_utils import ensure_windows_dlls, make_env


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Play an episode with a trained PPO model")
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 .zip model (e.g., q-supertuxkart/continuous/best_model.zip)")
    parser.add_argument("--stats", type=str, default=None, help="Optional VecNormalize stats path (vecnormalize.pkl)")
    parser.add_argument("--track", type=str, default="hacienda")
    parser.add_argument("--no-render", action="store_true", help="Run headless (no window) to validate setup")
    parsed = parser.parse_args(args=args)

    ensure_windows_dlls()
    # Preflight import of pystk2 so early DLL issues surface immediately
    import pystk2  # type: ignore

    # Create env for play with human rendering
    from pystk2_gymnasium import AgentSpec

    render_mode = None if parsed.no_render else "human"
    env = make_env(
        render_mode=render_mode,
        track=parsed.track,
        agent_spec=AgentSpec(use_ai=False),
        wrappers=None,
        vecnormalize_stats_path=parsed.stats,
        training=False,
    )

    # If VecNormalize was loaded, ensure eval mode
    try:
        from stable_baselines3.common.vec_env import VecNormalize

        if isinstance(env, VecNormalize):
            env.training = False
            env.norm_reward = False
            env.norm_obs_keys = ["continuous"]
    except Exception:
        pass

    model = PPO.load(parsed.model, env=env, device="auto")
    obs, info = env.reset(seed=0)

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

    env.close()
    print("Episode finished.")


if __name__ == "__main__":
    main()
