import gymnasium as gym
from pystk2_gymnasium import AgentSpec


# STK gymnasium uses one process
if __name__ == '__main__':
  # Use a a flattened version of the observation and action spaces
  # In both case, this corresponds to a dictionary with two keys:
  # - `continuous` is a vector corresponding to the continuous observations
  # - `discrete` is a vector (of integers) corresponding to discrete observations
  env = gym.make("supertuxkart/flattened-v0", render_mode="human", agent=AgentSpec(use_ai=False))

  ix = 0
  done = False
  state, *_ = env.reset()

  while not done:
      ix += 1
      action = env.action_space.sample()
      print(env)
      state, reward, terminated, truncated, _ = env.step(action)
      done = truncated or terminated

  # Important to stop the STK process
  env.close()