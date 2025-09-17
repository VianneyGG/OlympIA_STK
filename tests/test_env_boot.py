import os
import sys

import gymnasium as gym


def test_env_boot_once():
    # Basic smoke test: create env, reset, step, and close without rendering
    try:
        from pystk2_gymnasium import AgentSpec
    except Exception:
        # If the package is not importable, skip test
        import pytest

        pytest.skip("pystk2_gymnasium not available")

    env = gym.make(
        "supertuxkart/flattened_continuous_actions-v0",
        render_mode=None,
        agent=AgentSpec(use_ai=False),
        track="hacienda",
    )
    obs, info = env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
