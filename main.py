import os
from stable_baselines3 import PPO
from envs.gravity_env import GravityEnv
from visuals.simulation import run_inference

# =====================================================================
# CONFIGURATION FLAG
# =====================================================================
# Set to True if you want to train the stages sequentially.
# Set to False if you just want to run inference and generate videos.
TRAIN = False  
# =====================================================================

def run_curriculum_training(env):
    print("\n=========================================================")
    # 
    print("STARTING THREE-STAGE CURRICULUM TRAINING RUN")
    print("=========================================================")
    
    # -----------------------------------------------------------------
    # STAGE 1: LOCAL JOVIAN ORBIT INSERTION
    # -----------------------------------------------------------------
    print("\nExecuting Stage 1: Tightly Bound Local Capture...")
    env.max_curriculum_multiplier = 0.15
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,       # Dropped by an order of magnitude from 3e-4
        target_kl=0.015,          # Force early stopping if updates destabilize
        clip_range=0.15,          # Tighter clipping to prevent wild policy shifts
        verbose=1
    )
    model.learn(total_timesteps=100_000)
    model.save("ppo_gravity_stage1")
    print("Saved checkpoint: ppo_gravity_stage1.zip")

    # -----------------------------------------------------------------
    # STAGE 2: EXPANDING TO THE HILL SPHERE EDGE
    # -----------------------------------------------------------------
    print("\nExecuting Stage 2: Transition / Hill Sphere Boundary Zone...")
    env.max_curriculum_multiplier = 0.85
    model = PPO.load("ppo_gravity_stage1", env=env)

    import torch
    with torch.no_grad():
        # A value of -0.5 or 0.0 creates significant exploratory variance
        model.policy.log_std.fill_(-0.5)

    model.learn(total_timesteps=200_000)
    model.save("ppo_gravity_stage2")
    print("Saved checkpoint: ppo_gravity_stage2.zip")

    # -----------------------------------------------------------------
    # STAGE 3: DEEP SPACE APPROACH INTERCEPT
    # -----------------------------------------------------------------
    print("\nExecuting Stage 3: Long-Range Interplanetary Cruise...")
    env.max_curriculum_multiplier = 2.50
    model = PPO.load("ppo_gravity_stage2", env=env)

    import torch
    with torch.no_grad():
        # A value of -0.5 or 0.0 creates significant exploratory variance
        model.policy.log_std.fill_(-0.5)

    model.learn(total_timesteps=200_000)
    model.save("ppo_gravity_final")
    print("Saved final checkpoint: ppo_gravity_final.zip")
    print("\nTraining complete!")


def run_stage_visualizations(env):
    print("\n=========================================================")
    print("STARTING DETERMINISTIC INFERENCE VISUALIZATION")
    print("=========================================================")
    
    stages_to_test = {
        "Stage 3 (Deep Space Intercept)": {
            "model_path": "ppo_gravity_final.zip",
            "multiplier": 2.50,
            "output_video": "inference_stage3_final.mp4"
        }
    }

    """
            "Stage 1 (Local Orbit Capture)": {
            "model_path": "ppo_gravity_stage1.zip",
            "multiplier": 0.15,
            "output_video": "inference_stage1_local.mp4"

            "Stage 2 (Hill Sphere Boundary)": {
            "model_path": "ppo_gravity_stage2.zip",
            "multiplier": 1,
            "output_video": "inference_stage2_mid.mp4"
        },
    """
    
    for stage_name, config in stages_to_test.items():
        print(f"\nEvaluating: {stage_name}")
        
        # Guard clause to check if you've actually trained the checkpoint yet
        if not os.path.exists(config["model_path"]):
            print(f"⚠️ Checkpoint file '{config['model_path']}' not found. Run with TRAIN=True first.")
            continue
            
        # 1. Update curriculum dimensions for both the environment wrapper and simulation core
        env.max_curriculum_multiplier = config["multiplier"]
        env.sim_manager.curriculum_multiplier = config["multiplier"]
        
        # 2. Load the specific stage weight adjustments onto the env architecture
        model = PPO.load(config["model_path"], env=env, ent_coef=0.01)
        
        # 3. Call your native simulation inference method directly
        run_inference(model, env=env, filename=config["output_video"])


if __name__ == "__main__":
    # Initialize your shared custom environment wrapping your Rebound setup
    gravity_env = GravityEnv()
    
    if TRAIN:
        run_curriculum_training(gravity_env)
    else:
        run_stage_visualizations(gravity_env)