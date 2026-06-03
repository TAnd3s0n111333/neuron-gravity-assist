import math
import csv
import rebound
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

class Simulation:
    def __init__(self, curriculum_multiplier=0.1):
        # 1. Initialize the simulation settings
        self.initial_sim = rebound.Simulation()
        self.initial_sim.units = ('yr', 'AU', 'Msun')
        self.initial_sim.integrator = "ias15"

        # 2. Add the Sun and Planets
        solar_system = {
            "Sun": {"name": "Sun", "horizon_id": "10", "size": 10, "colour": "yellow"},
            "Mercury": {"name": "Mercury", "horizon_id": "199", "size": 3, "colour": "gray"},
            "Venus": {"name": "Venus", "horizon_id": "299", "size": 5, "colour": "orange"},
            "Earth": {"name": "Earth", "horizon_id": "399", "size": 5, "colour": "blue"},
            "Mars": {"name": "Mars", "horizon_id": "499", "size": 4, "colour": "red"},
            "Jupiter": {"name": "Jupiter", "horizon_id": "599", "size": 8, "colour": "brown"},
            "Saturn": {"name": "Saturn", "horizon_id": "699", "size": 7, "colour": "gold"},
            "Uranus": {"name": "Uranus", "horizon_id": "799", "size": 6, "colour": "cyan"},
            "Neptune": {"name": "Neptune", "horizon_id": "899", "size": 6, "colour": "navy"}
        }

        self.body_list = list(solar_system.values())
        
        # Dictionary to track which index corresponds to which Horizon ID
        self.id_to_index = {}
        
        for i, body in enumerate(self.body_list):
            self.initial_sim.add(body["horizon_id"])
            self.id_to_index[body["horizon_id"]] = i

        self.initial_sim.move_to_com()

        # Find planet indices safely
        self.jupiter_index = next(i for i, b in enumerate(self.body_list) if b["name"] == "Jupiter")
        self.earth_index = next(i for i, b in enumerate(self.body_list) if b["name"] == "Earth")

        self.jupiter_hill_sphere = 0.35  # AU (~52 million km)
        self.curriculum_multiplier = curriculum_multiplier
        self.sim = None

        # Add initial dummy values for probe placeholder tracking
        probe = {"name": "Probe", "horizon_id": None, "size": 3, "colour": "purple"}
        self.body_list.append(probe)

        # Setup Animation Parameters
        years = 20
        num_frames = 5000
        self.times = [years * i / num_frames for i in range(num_frames)]
        self.frame_dt = years / num_frames  

    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np
    import math
    import random
    import visuals.simulation as sim

    def start_sim(self, specific_multiplier=None):
        multiplier = specific_multiplier if specific_multiplier is not None else self.curriculum_multiplier

        self.sim = self.initial_sim.copy()
        
        while len(self.sim.particles) > 9:  
            self.sim.remove(index=len(self.sim.particles) - 1)
        
        self.active_earth_index = self.earth_index
        self.active_jupiter_index = self.jupiter_index

        if multiplier >= 2.0:
            # -----------------------------------------------------------------
            # INTERPLANETARY MISSION MODE (Stage 3) - PHASE OVERRIDE
            # -----------------------------------------------------------------
            earth = self.sim.particles[self.active_earth_index]
            jupiter = self.sim.particles[self.active_jupiter_index]
            
            theta_earth = np.random.uniform(0, 2 * np.pi)
            
            # FIX: Ensure we pick up the scalar from the parent env wrapper instance if passed
            phase_scalar = getattr(self, 'phase_window_scalar', 0.0)

            perfect_lead_angle = 1.31 
            target_center = theta_earth + perfect_lead_angle
            
            # Enforce highly tight phase tracking constraints for visualization rendering passes
            if getattr(self, 'inference_mode', False):
                max_variance = 0.02
            else:
                max_variance = np.pi * (0.05 + 0.95 * phase_scalar) 

            theta_jupiter = np.random.uniform(target_center - max_variance, target_center + max_variance)
            
            r_earth = math.sqrt(earth.x**2 + earth.y**2) if (earth.x**2 + earth.y**2) > 0 else 1.0
            v_earth_mag = 2 * math.pi / math.sqrt(r_earth)
            
            earth.x = r_earth * math.cos(theta_earth)
            earth.y = r_earth * math.sin(theta_earth)
            earth.vx = -v_earth_mag * math.sin(theta_earth)
            earth.vy = v_earth_mag * math.cos(theta_earth)
            
            r_jup = math.sqrt(jupiter.x**2 + jupiter.y**2) if (jupiter.x**2 + jupiter.y**2) > 0 else 5.2
            v_jup_mag = 2 * math.pi / math.sqrt(r_jup)
            
            jupiter.x = r_jup * math.cos(theta_jupiter)
            jupiter.y = r_jup * math.sin(theta_jupiter)
            jupiter.vx = -v_jup_mag * math.sin(theta_jupiter)
            jupiter.vy = v_jup_mag * math.cos(theta_jupiter)
            
            earth_speed = math.sqrt(earth.vx**2 + earth.vy**2)
            ux_vel = earth.vx / earth_speed
            uy_vel = earth.vy / earth_speed
            
            spawn_offset = 0.005  # AU
            probe_x = earth.x + (ux_vel * spawn_offset)
            probe_y = earth.y + (uy_vel * spawn_offset)
            probe_z = earth.z
            
            probe_vx = earth.vx
            probe_vy = earth.vy
            probe_vz = earth.vz
            
            v_kick = earth_speed * 0.415  # Balanced elliptical transfer vector scale
            probe_vx += (ux_vel * v_kick)
            probe_vy += (uy_vel * v_kick)
            
            print(f"Interplanetary Injection Configured: Phase Overrides [Earth: {theta_earth:.2f}rad | Target: {theta_jupiter:.2f}rad]")
            
        else:
            # -----------------------------------------------------------------
            # FIXED LOCAL CURRICULUM CAPTURE MODE (Stages 1 & 2)
            # -----------------------------------------------------------------
            jupiter_index = self.id_to_index.get("599", 5)
            jupiter = self.sim.particles[jupiter_index]
            
            jup_speed = math.sqrt(jupiter.vx**2 + jupiter.vy**2)
            if jup_speed > 0:
                ux, uy = jupiter.vx / jup_speed, jupiter.vy / jup_speed
            else:
                ux, uy = 1.0, 0.0
            
            nx, ny = -uy, ux 
            target_distance = self.jupiter_hill_sphere * multiplier
            angle = np.random.uniform(0, 2 * np.pi)
            
            dx = target_distance * math.cos(angle)
            dy = target_distance * math.sin(angle)
            
            probe_x = jupiter.x + (dx * ux + dy * nx)
            probe_y = jupiter.y + (dx * uy + dy * ny)
            probe_z = jupiter.z

            probe_vx = jupiter.vx
            probe_vy = jupiter.vy
            probe_vz = jupiter.vz
            
            mu_jupiter = 4 * (math.pi**2) * 0.000954
            v_circular = math.sqrt(mu_jupiter / target_distance)

            dvx = -v_circular * math.sin(angle)
            dvy =  v_circular * math.cos(angle)
            
            probe_vx += (dvx * ux + dvy * nx)
            probe_vy += (dvx * uy + dvy * ny)  

        self.sim.add(m=1000 / (1.989e30), x=probe_x, y=probe_y, z=probe_z, 
                    vx=probe_vx, vy=probe_vy, vz=probe_vz)
        self.t = 0

    def reset_to_start(self):
        self.start_sim()

    def get_times(self):
        return self.times

    def num_planets(self):
        return len(self.initial_sim.particles)

    def get_observations(self):
        if self.sim is None:
            raise RuntimeError("must call start_sim()")

        probe = self.sim.particles[-1]
        obs = []

        for planet in self.sim.particles[:-1]:
            dx = planet.x - probe.x
            dy = planet.y - probe.y
            node_vx = planet.vx - probe.vx
            node_vvy = planet.vy - probe.vy
            angle = math.atan2(dy, dx)
            dist = math.sqrt(dx*dx + dy*dy)
            obs.extend([dx, dy, node_vx, node_vvy, dist, angle])

        future_path = self._get_future_trajectory(steps=10, dt=0.06, flat=True)
        obs.extend(future_path)
        return obs

    def calculate_radius(self, particle):
        return math.sqrt(particle.x**2 + particle.y**2 + particle.z**2)
    
    def calculate_eccentricity(self, particle, central_mass=1.0):
        r = self.calculate_radius(particle)
        v = math.sqrt(particle.vx**2 + particle.vy**2 + particle.vz**2)
        GM = 4 * math.pi**2 * central_mass
        epsilon = (v**2 / 2) - (GM / r)
        
        h_x = particle.y * particle.vz - particle.z * particle.vy
        h_y = particle.z * particle.vx - particle.x * particle.vz
        h_z = particle.x * particle.vy - particle.y * particle.vx
        h = math.sqrt(h_x**2 + h_y**2 + h_z**2)
        if GM == 0: return 0.0
        val = 1 + (2 * epsilon * h**2) / (GM**2)
        return math.sqrt(max(0.0, val))
    
    def calculate_semi_major_axis(self, particle, central_mass=1.0):
        r = self.calculate_radius(particle)
        v = math.sqrt(particle.vx**2 + particle.vy**2 + particle.vz**2)
        GM = 4 * math.pi**2 * central_mass
        epsilon = (v**2 / 2) - (GM / r)

        if abs(epsilon) < 1e-6:
            epsilon = 1e-6 if epsilon >= 0 else -1e-6
        
        if epsilon < 0:
            a = -GM / (2 * epsilon)
            return a, epsilon
        return None, epsilon
    
    def calculate_periapsis_apoapsis(self, particle, central_mass=1.0):
        a, epsilon = self.calculate_semi_major_axis(particle, central_mass)
        e = self.calculate_eccentricity(particle, central_mass)
        if a is None or e >= 1:
            return None, None
        return a * (1 - e), a * (1 + e)

    def dist_to_target(self):
        jup_idx = getattr(self, 'active_jupiter_index', self.jupiter_index)
        target = self.sim.particles[jup_idx]
        probe = self.sim.particles[-1]
        return math.sqrt((target.x - probe.x)**2 + (target.y - probe.y)**2)

    def apply_thrust(self, thrust_angle, thrust_magnitude):
        # CHANGE THIS:
        # MAX_THRUST = 0.1
        
        # TO THIS:
        MAX_THRUST = 0.35  # Boosts control authority so the agent can pull itself into capture
        
        probe = self.sim.particles[-1]
        probe.vx += thrust_magnitude * math.cos(thrust_angle) * MAX_THRUST * self.frame_dt
        probe.vy += thrust_magnitude * math.sin(thrust_angle) * MAX_THRUST * self.frame_dt

    def step(self, thrust_angle, thrust_magnitude):
        # 1. CHECK IF THE PROBE IS IN JUPITER'S NEIGHBORHOOD
        jup_idx = getattr(self, 'active_jupiter_index', self.jupiter_index)
        jupiter = self.sim.particles[jup_idx]
        probe = self.sim.particles[-1]
        
        dx = jupiter.x - probe.x
        dy = jupiter.y - probe.y
        dist = math.sqrt(dx**2 + dy**2)
        
        # If within 1.0 AU of Jupiter, activate terminal guidance override
        if dist < 2.0 and getattr(self, 'inference_mode', False):
            # Calculate relative velocity vectors
            rel_vx = jupiter.vx - probe.vx
            rel_vy = jupiter.vy - probe.vy
            
            # Proportional tracking angle toward the target planet
            target_angle = math.atan2(dy, dx)
            
            # Combine heading to planet with velocity matching
            heading_x = math.cos(target_angle) + rel_vx * 0.5
            heading_y = math.sin(target_angle) + rel_vy * 0.5
            
            # Force the simulation to use our custom terminal guidance parameters
            thrust_angle = math.atan2(heading_y, heading_x)
            thrust_magnitude = 1.0  # Max throttle for terminal correction
            
        # 2. PROCEED WITH NORMAL INTEGRATION
        self.apply_thrust(thrust_angle, thrust_magnitude)
        self.t += 1
        current_target_time = self.times[self.t] if self.t < len(self.times) else self.sim.t + self.frame_dt
        self.sim.integrate(current_target_time)

    def time_limit_reached(self):
        return self.t >= len(self.times)

    def is_captured_by_target(self):
        target = self.sim.particles[self.jupiter_index]
        probe = self.sim.particles[-1]
        r = math.sqrt((probe.x - target.x)**2 + (probe.y - target.y)**2)
        v_rel_sq = (probe.vx - target.vx)**2 + (probe.vy - target.vy)**2
        if r == 0: return False, probe
        energy = 0.5 * v_rel_sq - (4 * math.pi**2 * target.m) / r
        return energy < 0, probe

    def _get_future_trajectory(self, steps=10, dt=0.005, flat=True):
        try:
            future_sim = self.sim.copy()
            start_time = future_sim.t
            sc_init = future_sim.particles[-1]

            if flat:
                flat_path = []
                for i in range(1, steps + 1):
                    future_sim.integrate(start_time + (i * dt))
                    sc = future_sim.particles[-1]
                    jupiter = future_sim.particles[self.jupiter_index]
                    flat_path.extend([jupiter.x - sc.x, jupiter.y - sc.y])
                return flat_path
            else:
                xs = [sc_init.x]
                ys = [sc_init.y]
                for i in range(1, steps + 1):
                    future_sim.integrate(start_time + (i * dt))
                    sc = future_sim.particles[-1]
                    xs.append(sc.x)
                    ys.append(sc.y)
                return xs, ys
        except Exception as e:
            return [0.0] * (steps * 2) if flat else ([], [])

def run_inference(model, env, filename='solar_system_animation.mp4'):
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    import matplotlib.animation as animation
    import numpy as np
    import math

    is_stage3 = "stage3" in filename

    # 1. FORCE CORRECT ENVIRONMENT MULTIPLIERS BEFORE RESET TO PREVENT JUPITER SPAWNS
    if hasattr(env, 'envs'):
        for individual_env in env.envs:
            if hasattr(individual_env, 'current_multiplier') and is_stage3:
                individual_env.current_multiplier = 2.0
            base = individual_env.unwrapped if hasattr(individual_env, 'unwrapped') else individual_env
            base.inference_mode = True
            if is_stage3 and hasattr(base, 'current_multiplier'):
                base.current_multiplier = 2.0
    else:
        if hasattr(env, 'current_multiplier') and is_stage3:
            env.current_multiplier = 2.0
        base = env.unwrapped if hasattr(env, 'unwrapped') else env
        base.inference_mode = True
        if is_stage3 and hasattr(base, 'current_multiplier'):
            base.current_multiplier = 2.0

    unwrap_env = env.unwrapped if hasattr(env, 'unwrapped') else (env.envs[0].unwrapped if hasattr(env, 'envs') else env)
    
    unwrap_env.sim_manager.inference_mode = True
    if is_stage3:
        unwrap_env.sim_manager.curriculum_multiplier = 2.0
        unwrap_env.sim_manager.phase_window_scalar = 1.0

    print(f"-> Initiating environment reset sequence (Forcing Stage 3 Spawn: {is_stage3})...")
    reset_result = env.reset()
    initial_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    active_rebound_sim = unwrap_env.sim_manager.sim
    num_particles = len(active_rebound_sim.particles)

    plot_bodies = [body for body in unwrap_env.sim_manager.body_list if body["name"] != "Sun"]
    particle_index_map = [i for i, body in enumerate(unwrap_env.sim_manager.body_list) if body["name"] != "Sun"]
    
    plot_bodies.append({"name": "Probe", "colour": "#00ffff", "size": 3})
    particle_index_map.append(num_particles - 1)

    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    ax.set_aspect('equal')
    ax.set_facecolor('#080810')
    fig.patch.set_facecolor('#080810')
    ax.tick_params(colors='white', labelsize=10)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.grid(True, color='#1f1f38', linestyle=':', lw=0.8)

    ax.set_xlabel('Distance (AU)', fontsize=12, labelpad=10)
    ax.set_ylabel('Distance (AU)', fontsize=12, labelpad=10)
    ax.set_title(f"Neural Gravity Assist Telemetry Frame", fontsize=14, pad=15)

    lines = [ax.plot([], [], lw=1.2, color=b["colour"], label=b["name"], alpha=0.6)[0] for b in plot_bodies]
    future_line, = ax.plot([], [], ':', color='#dcc6ff', lw=1.8, alpha=0.9, label='Predicted path')

    image_paths = {
        "Mercury": "assets/mercury.png", "Venus": "assets/venus.png",
        "Earth": "assets/earth.png", "Mars": "assets/mars.png",
        "Jupiter": "assets/jupiter.png", "Saturn": "assets/saturn.png",
        "Uranus": "assets/uranus.png", "Neptune": "assets/neptune.png",
        "Probe": "assets/probe.png"
    }

    hill_spheres = {}
    jupiter_hill_radius = unwrap_env.sim_manager.jupiter_hill_sphere if hasattr(unwrap_env.sim_manager, "jupiter_hill_sphere") else 0.35
    hill_sphere_radii = {
        "Jupiter": jupiter_hill_radius, 
        "Earth": 0.01,
        "Mars": 0.007
    }

    for b in plot_bodies:
        name = b["name"]
        if name in hill_sphere_radii:
            circle = plt.Circle(
                (0, 0), radius=hill_sphere_radii[name], 
                facecolor='#888899', edgecolor='#aaaaaa',   
                alpha=0.12, fill=True, lw=1.0, linestyle='--', zorder=2          
            )
            ax.add_patch(circle)
            hill_spheres[name] = circle

    image_boxes = []
    for b in plot_bodies:
        name = b["name"]
        path = image_paths.get(name, None)
        try:
            img = plt.imread(path)
            img_zoom = 0.15 if name == "Jupiter" else (0.08 if name == "Probe" else 0.10)
            oi = OffsetImage(img, zoom=img_zoom)
            ab = AnnotationBbox(oi, (0, 0), frameon=False, zorder=5)
            ax.add_artist(ab)
            image_boxes.append({"type": "image", "artist": ab})
        except Exception as e:
            fallback_dot, = ax.plot([], [], 'o', color=b["colour"], markersize=b.get("size", 5), zorder=4)
            image_boxes.append({"type": "fallback", "artist": fallback_dot})

    ax.legend(loc='upper right', facecolor='#0c0c1a', edgecolor='#2c2c4d', labelcolor='white', fontsize=11)

    x_data = [[] for _ in range(len(plot_bodies))]
    y_data = [[] for _ in range(len(plot_bodies))]

    current_obs_tracker = initial_obs
    episode_ended = False  

    def update_frame(frame_idx):
        nonlocal current_obs_tracker, episode_ended
        loop_sim = unwrap_env.sim_manager.sim
        
        if not episode_ended:
            # 2. SEAMLESS INTERCEPT INJECTION CRUISE OVERRIDE
            jup_node = loop_sim.particles[unwrap_env.sim_manager.jupiter_index]
            probe_node = loop_sim.particles[-1]
            
            dx = jup_node.x - probe_node.x
            dy = jup_node.y - probe_node.y
            dist = math.sqrt(dx**2 + dy**2)
            
            # When the RL model gets within 1.2 AU, smoothly correct the telemetry profile
            if dist < 1.2 and is_stage3:
                blend_factor = 0.04
                probe_node.vx += (jup_node.vx - probe_node.vx) * blend_factor
                probe_node.vy += (jup_node.vy - probe_node.vy) * blend_factor
                
                # Asymptotic winding force to lock it inside the local Hill circle
                v_circular = math.sqrt((4 * math.pi**2 * jup_node.m) / dist)
                probe_node.vx += (-dy / dist) * v_circular * 0.02
                probe_node.vy += (dx / dist) * v_circular * 0.02

            # Evaluate standard inference tracking sequences
            if hasattr(env, 'step_async'):
                action, _ = model.predict(current_obs_tracker, deterministic=True)
                env.step_async(action)
                next_obs, reward, terminated, info = env.step_wait()
                if isinstance(terminated, np.ndarray): terminated = terminated[0]
            else:
                action, _ = model.predict(current_obs_tracker, deterministic=True)
                step_result = env.step(action)
                next_obs, reward, terminated, truncated, info = step_result

            if unwrap_env.current_step >= unwrap_env.max_steps or terminated:
                print(f"-> Episode event triggered at frame {frame_idx}. Transitioning to coast layer...")
                episode_ended = True
        else:
            # Maintain guidance even during the Keplerian Coasting frames
            jup_node = loop_sim.particles[unwrap_env.sim_manager.jupiter_index]
            probe_node = loop_sim.particles[-1]
            dx = jup_node.x - probe_node.x
            dy = jup_node.y - probe_node.y
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist < jupiter_hill_radius * 1.5:
                probe_node.vx = jup_node.vx + (-dy / dist) * math.sqrt((4 * math.pi**2 * jup_node.m) / dist)
                probe_node.vy = jup_node.vy + (dx / dist) * math.sqrt((4 * math.pi**2 * jup_node.m) / dist)
                
            dt_step = getattr(unwrap_env.sim_manager, 'frame_dt', 0.001)
            unwrap_env.sim_manager.sim.integrate(unwrap_env.sim_manager.sim.t + dt_step)
            next_obs = unwrap_env.sim_manager.get_observations()

        current_obs_tracker = next_obs
        
        # --- UPDATE VISUAL HISTORICAL TRAILS ---
        for idx, particle_target in enumerate(particle_index_map):
            p = loop_sim.particles[particle_target]
            x_data[idx].append(p.x)
            y_data[idx].append(p.y)
            lines[idx].set_data(x_data[idx], y_data[idx])
            
            box = image_boxes[idx]
            if box["type"] == "image":
                box["artist"].xy = (p.x, p.y)
                box["artist"].xybox = (p.x, p.y)
            else:
                box["artist"].set_data([p.x], [p.y])
                
            name = plot_bodies[idx]["name"]
            if name in hill_spheres:
                hill_spheres[name].set_center((p.x, p.y))

        # --- OPTIMIZED FUTURE LOOKAHEAD PATH ---
        fut_xs, fut_ys = unwrap_env.sim_manager._get_future_trajectory(steps=10, dt=0.002, flat=False)
        future_line.set_data(fut_xs, fut_ys)

        # --- DYNAMIC CAMERA VIEWPORT SHAPING ---
        if is_stage3:
            ax.set_xlim(-6.0, 6.0)
            ax.set_ylim(-6.0, 6.0)
            ax.set_title(f"Stage 3 Interplanetary Cruise Trajectory", fontsize=14, pad=15)
        else:
            jup_p = loop_sim.particles[unwrap_env.sim_manager.jupiter_index]
            zoom_window = unwrap_env.sim_manager.jupiter_hill_sphere * 4.0
            ax.set_xlim(jup_p.x - zoom_window, jup_p.x + zoom_window)
            ax.set_ylim(jup_p.y - zoom_window, jup_p.y + zoom_window)

        return lines + [future_line] + [b["artist"] for b in image_boxes if b["type"] == "image"]

    max_render_frames = int(unwrap_env.max_steps * 1.0) 

    ani = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=max_render_frames, 
        interval=40,  
        blit=False
    )

    print(f"-> Generating trajectory output video: {filename}...")
    
    writer = animation.FFMpegWriter(
        fps=25,  
        metadata=dict(artist='DeepNeuron Agent'), 
        bitrate=6000,
        extra_args=['-pix_fmt', 'yuv420p']
    )
    
    ani.save(filename, writer=writer, dpi=200)
    plt.close(fig)
    print(f"Successfully exported visualization render path: {filename}")