import os
import sys
from pathlib import Path

# Ensure Windows can find native DLLs for pystk2
env_bin = Path(sys.prefix) / "Library" / "bin"
if env_bin.is_dir():
  try:
    os.add_dll_directory(str(env_bin))
  except Exception:
    pass

# Add common SuperTuxKart install locations (optional but helps on Windows)
for stk_dir in [
  Path(r"C:\Program Files\SuperTuxKart 1.4"),
  Path(r"C:\Program Files\SuperTuxKart"),
]:
  if stk_dir.is_dir():
    try:
      os.add_dll_directory(str(stk_dir))
    except Exception:
      pass

import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from stable_baselines3 import PPO


# STK gymnasium uses one process
if __name__ == '__main__':
    # Use a a flattened version of the observation and action spaces
    # In both case, this corresponds to a dictionary with two keys:
    # - `continuous` is a vector corresponding to the continuous observations
    # - `discrete` is a vector (of integers) corresponding to discrete observations
    env = gym.make("supertuxkart/flattened_discrete-v0", render_mode="human", agent=AgentSpec(use_ai=False), track="hacienda")

    model = PPO.load("ppo_stk.zip", env=env)
    ix = 0
    done = False
    state, *_ = env.reset()

    while not done:
        ix += 1
        action = model.predict(state, deterministic=True)[0]
        state, reward, terminated, truncated, _ = env.step(action)
        done = truncated or terminated

    # Important to stop the STK process
    env.close()