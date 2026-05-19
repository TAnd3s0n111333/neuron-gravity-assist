import rebound
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# 1. Initialize the simulation
sim = rebound.Simulation()
sim.units = ('yr', 'AU', 'Msun')
sim.integrator = "ias15"


#TODO: 
# 1.Cretae the particle -> 
# 2. make it be able to move - controls be a list of inputs 
# 3. Save Info - position, velocity, acceleration
# 2. Add the Sun and Planets

# Exact physical body center IDs (avoids mixing up planet-moon barycenters)
# planets = ["10", "199", "299", "399", "499", "599", "699", "799", "899"] #TODO: Maybe dict and add size
planets = ["10", "199", "299", "399", "499"] #Upto mars for now

sim.add(m=1/(1.989e-30)*1000, x=0.0, y=0.0, z=0.0)
earth_x = sim.particles["399"].x
earth_y = sim.particles["399"].y
earth_z = sim.particles["399"].z

# planets = ["Sun", "Mercury", "Venus", "Earth", "Mars"] #TODO: Add all planets
for planet in planets:
    sim.add(planet)
# sim.add("301") 
sim.move_to_com()

# 3. Setup Animation Parameters
years = 10
num_frames = 1000
times = [years * i / num_frames for i in range(num_frames)]

# Setup the figure and axes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_xlabel('Distance (AU)')
ax.set_ylabel('Distance (AU)')
ax.set_title('Inner Solar System Animation')

# Create plot elements for each planet: (line for trail, dot for planet)
lines = [ax.plot([], [], lw=1, label=planets[i])[0] for i in range(len(planets))]

# Define colors to match your planet order
planet_colors = ['#FFD700', '#808080', '#E6E6FA', '#1E90FF', '#FF4500'] # Sun, Merc, Ven, Earth, Mars
planet_sizes  = [16,        5,         7,         8,         6]
scale = 3389/5
# planet_sizes = [696340/scale, 2439/scale, 6051/scale, 6371/scale, 3389/scale]

dots = [
    ax.plot([], [], 'o', ms=planet_sizes[i], color=planet_colors[i])[0] 
    for i in range(len(planets))
]
# dots = [ax.plot([], [], 'o')[0] for i in range(len(planets))]
ax.legend(loc='upper right')

# Data storage for trails
x_data = [[] for _ in range(len(planets))]
y_data = [[] for _ in range(len(planets))]

# 4. Animation Function
def update(frame):
    sim.integrate(times[frame])
    
    for i, p in enumerate(sim.particles):
        # Update trail data
        x_data[i].append(p.x)
        y_data[i].append(p.y)
        
        # Update the visual elements
        lines[i].set_data(x_data[i], y_data[i])
        dots[i].set_data([p.x], [p.y])
        
    return lines + dots

# 5. Run and Save Animation
ani = FuncAnimation(fig, update, frames=num_frames, interval=30, blit=True)

# To save as MP4 (requires ffmpeg) or GIF (requires pillow)
ani.save('solar_system_animation.mp4', writer='ffmpeg', fps=30)

print("Animation saved as solar_system_animation.mp4")
plt.show() # Uncomment to view while running



sim.add(m=0, x=1.005, y=0.0, z=0.0, vx=0, vy=6.20, hash="Probe")

# ----------------------------------------------------
# 4. Define your Thrust Inputs
# ----------------------------------------------------
# Control vector: [Thrust along X, Thrust along Y, Thrust along Z]
# Units here scale with AU/yr^2. Keep these values very small!
active_thrust_input = np.array([0.001, 0.0, 0.0]) 

def engine_controller(sim_pointer):
    """
    This function intercepts REBOUND's core loops at every time step 
    and applies constant force directly to the probe particle.
    """
    # Locating our probe inside the simulation memory matrix
    probe = sim_pointer.contents.particles["Probe"]
    
    # Apply our custom acceleration to the probe's velocity derivatives
    probe.ax += active_thrust_input[0]
    probe.ay += active_thrust_input[1]
    probe.az += active_thrust_input[2]

# 5. Lock the controller into the simulation heartbeat
sim.additional_forces = engine_controller
sim.force_is_constant = 1 # Tells REBOUND the force changes smoothly