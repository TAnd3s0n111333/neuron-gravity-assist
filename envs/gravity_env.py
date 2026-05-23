import gymnasium as gym
from gymnasium import spaces
import numpy as np

import visuals.simulation as sim

class GravityEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.sim = sim.Simulation()

        obs_size = 6 * self.sim.num_planets()  # per planet: dx, dy, dvx, dvy, dist, angle

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: [thrust_angle (-pi to pi), thrust_magnitude (0 to 1.0)]
        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0]),
            high=np.array([np.pi, 1.0]),
            dtype=np.float32
        )

        self._prev_dist_to_target = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.start_sim()
        self._prev_dist_to_target = self.sim.dist_to_target()
        return np.array(self.sim.get_observations(), dtype=np.float32), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        angle, magnitude = action

        self.sim.step(angle, magnitude)

        terminated = self.sim.is_captured_by_target()
        truncated = self.sim.time_limit_reached()

        dist = self.sim.dist_to_target()
        reward = self._prev_dist_to_target - dist  # positive when getting closer

        if terminated:
            reward += 1000.0

        self._prev_dist_to_target = dist

        return np.array(self.sim.get_observations(), dtype=np.float32), reward, terminated, truncated, {}