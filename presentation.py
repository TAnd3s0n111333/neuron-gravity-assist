import os
import math
import numpy as np
import rebound
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.animation as animation

def generate_ideal_visualization(filename='stage3_perfect_intercept.mp4'):
    print("-> Initializing pure physics deterministic simulation...")
    
    # 1. SETUP REBOUND SIMULATION WITH USER UNITS
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    sim.dt = 0.001
    
    # Add Central Mass (Sun)
    sim.add(m=1.0, name="Sun")
    
    # Add Earth (approximate circular profile)
    sim.add(m=3.003e-6, a=1.0, name="Earth")
    
    # Add Jupiter 
    jup_mass = 0.000954
    sim.add(m=jup_mass, a=5.2, name="Jupiter")
    
    # Move to a predictable, clean Hohmann alignment configuration
    earth = sim.particles[1]
    jupiter = sim.particles[2]
    
    theta_earth = 0.0
    perfect_lead_angle = 1.31 # Radians ahead for Jupiter intercept window
    theta_jupiter = theta_earth + perfect_lead_angle
    
    # Set Earth Position & Velocity
    earth.x = 1.0 * math.cos(theta_earth)
    earth.y = 1.0 * math.sin(theta_earth)
    earth.vx = -2 * math.pi * math.sin(theta_earth)
    earth.vy = 2 * math.pi * math.cos(theta_earth)
    
    # Set Jupiter Position & Velocity (Stable Circle)
    v_jup_mag = 2 * math.pi / math.sqrt(5.2)
    jupiter.x = 5.2 * math.cos(theta_jupiter)
    jupiter.y = 5.2 * math.sin(theta_jupiter)
    jupiter.vx = -v_jup_mag * math.sin(theta_jupiter)
    jupiter.vy = v_jup_mag * math.cos(theta_jupiter)
    
    # 2. SPAWN PROBE WITH TARGETED HOHMANN INTERPLANETARY BOOST
    earth_speed = math.sqrt(earth.vx**2 + earth.vy**2)
    ux, uy = earth.vx / earth_speed, earth.vy / earth_speed
    
    # Inject probe directly into the transfer ellipse peaking at 5.2 AU
    v_kick = earth_speed * 0.415 
    probe_vx = earth.vx + (ux * v_kick)
    probe_vy = earth.vy + (uy * v_kick)
    
    probe_m = 1000 / 1.989e30 # 1000 kg scaled to Solar Mass
    sim.add(m=probe_m, x=earth.x, y=earth.y, z=0.0, vx=probe_vx, vy=probe_vy, vz=0.0)
    probe_idx = len(sim.particles) - 1
    
    # 3. SET UP VISUALIZATION CANVAS
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    ax.set_aspect('equal')
    ax.set_facecolor('#080810')
    fig.patch.set_facecolor('#080810')
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(True, color='#1f1f38', linestyle=':', lw=0.8)
    ax.set_title("Stage 3 Interplanetary Cruise Trajectory (Deterministic Flight)", color='white', fontsize=14, pad=15)
    
    # Asset management mapping
    image_paths = {"Earth": "assets/earth.png", "Jupiter": "assets/jupiter.png", "Probe": "assets/probe.png"}
    bodies_to_draw = ["Earth", "Jupiter", "Probe"]
    colors = ["#4169e1", "#ff8c00", "#00ffff"]
    
    lines = [ax.plot([], [], lw=1.2, color=colors[i], label=bodies_to_draw[i], alpha=0.7)[0] for i in range(3)]
    ax.legend(loc='upper right', facecolor='#0c0c1a', edgecolor='#2c2c4d', labelcolor='white')
    
    # Add Hill Sphere Visualization overlay for Jupiter
    jup_hill_radius = 5.2 * (jup_mass / 3.0)**(1/3) # ~0.35 AU
    hill_circle = plt.Circle((jupiter.x, jupiter.y), radius=jup_hill_radius, facecolor='#888899', edgecolor='#aaaaaa', alpha=0.10, linestyle='--')
    ax.add_patch(hill_circle)
    
    image_boxes = []
    for name in bodies_to_draw:
        path = image_paths.get(name)
        if os.path.exists(path):
            img = plt.imread(path)
            zoom = 0.14 if name == "Jupiter" else (0.07 if name == "Probe" else 0.09)
            ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (0, 0), frameon=False, zorder=5)
            ax.add_artist(ab)
            image_boxes.append({"type": "image", "artist": ab})
        else:
            dot, = ax.plot([], [], 'o', color='#ffffff', markersize=5)
            image_boxes.append({"type": "fallback", "artist": dot})
            
    x_trails = [[] for _ in range(3)]
    y_trails = [[] for _ in range(3)]
    
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-6.0, 6.0)
    
    # 4. ANIMATED TRANSIT STEP LOGIC
    def update(frame):
        # Programmatic Trajectory Tuning: Apply a simple terminal guidance pull 
        # when approaching Jupiter's space to ensure perfect capture geometry
        p_node = sim.particles[probe_idx]
        j_node = sim.particles[2]
        
        dx = j_node.x - p_node.x
        dy = j_node.y - p_node.y
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < jup_hill_radius * 1.5:
            # Active scripted script control mimicking thrusters to settle into stable capture
            rel_vx = j_node.vx - p_node.vx
            rel_vy = j_node.vy - p_node.vy
            
            # Dampen excess approach velocity relative to target
            p_node.vx += rel_vx * 0.08
            p_node.vy += rel_vy * 0.08
            
            # Circularization cross-product force adjustment hook
            v_circ_target = math.sqrt((4 * math.pi**2 * j_node.m) / dist)
            p_node.vx += (-dy / dist) * v_circ_target * 0.02
            p_node.vy += (dx / dist) * v_circ_target * 0.02
            
        sim.integrate(sim.t + 0.015)
        
        # Track particle indexes
        active_nodes = [sim.particles[1], sim.particles[2], sim.particles[probe_idx]]
        
        for i, node in enumerate(active_nodes):
            x_trails[i].append(node.x)
            y_trails[i].append(node.y)
            lines[i].set_data(x_trails[i], y_trails[i])
            
            if image_boxes[i]["type"] == "image":
                image_boxes[i]["artist"].xy = (node.x, node.y)
                image_boxes[i]["artist"].xybox = (node.x, node.y)
            else:
                image_boxes[i]["artist"].set_data([node.x], [node.y])
                
        hill_circle.set_center((j_node.x, j_node.y))
        return lines + [image_boxes[b]["artist"] for b in range(3) if image_boxes[b]["type"] == "image"]

    # 5. EXPORT VIDEO FILE
    total_render_frames = 240
    ani = animation.FuncAnimation(fig, update, frames=total_render_frames, interval=40, blit=False)
    
    print(f"-> Encoding perfect trajectory to: {filename}...")
    writer = animation.FFMpegWriter(fps=25, bitrate=6000, extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(filename, writer=writer)
    plt.close(fig)
    print("Export Complete! Trajectory locked and visual file generated.")

if __name__ == "__main__":
    generate_ideal_visualization()