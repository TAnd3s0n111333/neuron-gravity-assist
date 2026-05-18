import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

class CustomEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define action_space: Discrete or Box (Continuous)
        self.action_space = spaces.Discrete(2)
        # Define observation_space: Use Box for numerical ranges
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.observation_space.sample()
        return observation, {}

    def step(self, action):
        # Your logic here
        observation = self.observation_space.sample()
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

# Instantiate the environment
env = CustomEnv()

# Link the model to your environment
model = PPO("MlpPolicy", env, verbose=1)

# Start training
model.learn(total_timesteps=10000)

# Save the trained agent
model.save("ppo_custom_env")