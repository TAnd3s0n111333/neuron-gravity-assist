import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import random
import visuals.simulation as sim

class GravityEnv(gym.Env):
    @property
    def sim(self):
        """Property alias to preserve backward compatibility with the visualization module hooks"""
        return self.sim_manager
    def __init__(self):
        super().__init__()
        # Rename to self.sim_manager to prevent variable clobbering with raw REBOUND classes
        self.sim_manager = sim.Simulation(curriculum_multiplier=0.1)
        self.total_env_timesteps = 0 
        self.sim_manager.start_sim() 

        self.max_steps = 1000
        self.inference_mode = False  
        
        sample_obs = self.sim_manager.get_observations()
        obs_size = len(sample_obs) 
        
        print(f"--> SYSTEM CHECK: Initializing observation space with size: {obs_size}")

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0], dtype=np.float32),
            high=np.array([np.pi, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self._prev_dist_to_target = None
        self.max_curriculum_multiplier = 0.15

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Handle curriculum levels
        if self.max_curriculum_multiplier >= 2.0:
            if random.random() < 0.20:
                self.current_multiplier = random.uniform(0.10, 0.50)  
            else:
                self.current_multiplier = 2.0  
        else:
            if random.random() < 0.20:
                self.current_multiplier = random.uniform(0.05, 0.15)
            else:
                self.current_multiplier = random.uniform(0.10, self.max_curriculum_multiplier)
            
        # Set phase window alignment scalars
        if self.inference_mode:
            self.sim_manager.phase_window_scalar = 1.0
        else:
            total_steps = getattr(self, 'total_env_timesteps', 0)
            self.sim_manager.phase_window_scalar = min(1.0, (total_steps / 1_000_000))

        # =================================================================
        # FIXES IMPLEMENTED: 
        # 1. No longer overwriting self.sim with an empty raw REBOUND instance.
        # 2. Executing start_sim securely directly through your simulation wrapper framework.
        # =================================================================
        self.sim_manager.start_sim(specific_multiplier=self.current_multiplier)
        
        self.current_step = 0
        
        # Dynamically set max steps per episode based on stage duration
        if self.current_multiplier >= 2.0:
            self.max_steps = 5000  
        else:
            self.max_steps = int(1000 * max(1.0, self.current_multiplier))

        # Compute initial tracking metrics relative to Jupiter target via the wrapper manager
        initial_dist = self.sim_manager.dist_to_target()
        self._prev_dist_to_target = initial_dist
        self.was_in_hill_sphere = (initial_dist <= self.sim_manager.jupiter_hill_sphere)
        
        # Return observation and info dictionary according to Gymnasium standard
        return np.array(self.sim_manager.get_observations(), dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1
        self.total_env_timesteps += 1  
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        angle, magnitude = action

        # Core execution steps passed down safely to your engine
        self.sim_manager.step(angle, magnitude)

        # 1. Integrator Sanitization Layer
        raw_obs = self.sim_manager.get_observations()
        if np.isnan(raw_obs).any() or np.isinf(raw_obs).any():
            print("Integrator instability detected! Forcing defensive termination.")
            cleaned_obs = np.nan_to_num(raw_obs, nan=0.0, posinf=10.0, neginf=-10.0)
            return cleaned_obs, -10.0, True, False, {}

        dist = self.sim_manager.dist_to_target()
        
        jupiter_index = self.sim_manager.id_to_index.get("599", 5)
        target = self.sim_manager.sim.particles[jupiter_index] 
        probe = self.sim_manager.sim.particles[-1]

        eccentricity = self.sim_manager.calculate_eccentricity(probe)
        periapsis, apoapsis = self.sim_manager.calculate_periapsis_apoapsis(probe)
        semi_major_axis, epsilon = self.sim_manager.calculate_semi_major_axis(probe)
        
        current_in_hill_sphere = (dist <= self.sim_manager.jupiter_hill_sphere)

        # 2. Heading Velocity Calculation
        dx_rel = target.x - probe.x
        dy_rel = target.y - probe.y
        dist_3d = math.sqrt(dx_rel**2 + dy_rel**2)

        dvx_rel = probe.vx - target.vx
        dvy_rel = probe.vy - target.vy

        if dist_3d > 0:
            dx_unit = dx_rel / dist_3d
            dy_unit = dy_rel / dist_3d
            approach_velocity = (dvx_rel * dx_unit) + (dvy_rel * dy_unit)
        else:
            approach_velocity = 0

        # 3. Standard Episode Bounds
        is_captured, _ = self.sim_manager.is_captured_by_target()
        
        if getattr(self, 'inference_mode', False):
            terminated = False
            truncated = False
        else:
            terminated = is_captured
            truncated = (self.current_step >= self.max_steps)

        # --- REWARD SHAPING & TERMINATION CURRICULUM ---
        if self.current_multiplier >= 2.0:
            # -----------------------------------------------------------------
            # STAGE 3: TRUE INTERPLANETARY CRUISE SHAPING
            # -----------------------------------------------------------------
            prev_dist = self._prev_dist_to_target if self._prev_dist_to_target is not None else dist
            distance_drop = prev_dist - dist
            
            reward = (10.0 * distance_drop) - (0.005 * magnitude)
            
            sun = self.sim_manager.sim.particles[0]
            dist_to_sun = math.sqrt((probe.x - sun.x)**2 + (probe.y - sun.y)**2)
            
            if dist_to_sun < 5.2:
                reward += 2.0 * (dist_to_sun - 1.0)
            
            if dist > self.sim_manager.jupiter_hill_sphere:
                if truncated:
                    reward -= 5.0
                    print("Stage 3 Time Limit Reached")
                elif dist > 150.0:
                    reward -= 10.0
                    terminated = True
                    print("Probe Lost to Deep Interstellar Space")
            else:
                reward += 50.0  # Milestone bonus
                
                relative_speed = math.sqrt((probe.vx - target.vx)**2 + (probe.vy - target.vy)**2)
                reward += 10.0 * (1.0 / (1.0 + relative_speed))  
                
                if epsilon < 0 and 0 <= eccentricity <= 1:
                    reward += 5000.0  # Bound Capture Bonus
                    if not getattr(self, 'inference_mode', False):
                        terminated = True  
                        print("SUCCESS: Full Interplanetary Insertion and Capture Executed!")
                else:
                    if eccentricity > 1:
                        reward -= (eccentricity - 1.0) * 5.0
                    
                if self.was_in_hill_sphere and not current_in_hill_sphere:
                    reward -= 20.0
                    if not getattr(self, 'inference_mode', False):
                        terminated = True
                        print("FAILURE: Probe overshot target well into deep space.")
        else:
            # -----------------------------------------------------------------
            # STAGES 1 & 2: LOCAL CURRICULUM CAPTURE MODE
            # -----------------------------------------------------------------
            reward = (1 / (1 + dist)) - (0.005 * magnitude)

            if truncated:
                reward -= 2.0
                print("Time Limit Reached")
            elif dist > 150.0: 
                reward -= 5.0
                terminated = True
                print("Probe Escaped Solar System")
            elif terminated:
                reward += 100.0  
            elif self.was_in_hill_sphere and not current_in_hill_sphere:
                reward -= 10.0
                terminated = True  
                print("FAILURE: Probe abandoned Jupiter after approach. Terminating.")
            else:
                if dist > self.sim_manager.jupiter_hill_sphere:
                    reward += 1.0 * approach_velocity  
                    distance_to_hill = dist - self.sim_manager.jupiter_hill_sphere
                    reward += 2.0 * math.exp(-2.0 * distance_to_hill)
                else:
                    reward += 20.0  
                    relative_speed = math.sqrt((probe.vx - target.vx)**2 + (probe.vy - target.vy)**2)
                    reward += 5.0 * (1.0 / (1.0 + relative_speed))  
                    
                    if epsilon < 0 and 0 <= eccentricity <= 1:
                        reward += 10.0 * (1.0 - eccentricity)

        # 5. Persistent State Preservation
        self.was_in_hill_sphere = current_in_hill_sphere
        self._prev_dist_to_target = dist

        return np.array(raw_obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), {}