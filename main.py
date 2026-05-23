from envs.gravity_env import GravityEnv
from stable_baselines3 import PPO
from visuals.simulation import Simulation

env = GravityEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("gravity_assist_ppo")

sim = Simulation()
sim.run_inference(model)