import rebound
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import csv

# 1. Initialize the simulation
sim = rebound.Simulation()
sim.units = ('yr', 'AU', 'Msun')
sim.integrator = "whfast"

# 2. Add the Sun and Planets
solar_system = {
    "Sun": {
        "name": "Sun",
        "horizon_id": "10",
        "size": 10,
        "colour": "yellow"
    },
    "Mercury": {
        "name": "Mercury",
        "horizon_id": "199",
        "size": 3,
        "colour": "gray"
    },
    "Venus": {
        "name": "Venus",
        "horizon_id": "299",
        "size": 5,
        "colour": "orange"
    },
    "Earth": {
        "name": "Earth",
        "horizon_id": "399",
        "size": 5,
        "colour": "blue"
    },
    "Mars": {
        "name": "Mars",
        "horizon_id": "499",
        "size": 4,
        "colour": "red"
    },
    "Jupiter": {
        "name": "Jupiter",
        "horizon_id": "599",
        "size": 8,
        "colour": "brown"
    },
    "Saturn": {
        "name": "Saturn",
        "horizon_id": "699",
        "size": 7,
        "colour": "gold"
    },
    "Uranus": {
        "name": "Uranus",
        "horizon_id": "799",
        "size": 6,
        "colour": "cyan"
    },
    "Neptune": {
        "name": "Neptune",
        "horizon_id": "899",
        "size": 6,
        "colour": "navy"
    }
}

body_list = list(solar_system.values())
for body in body_list:
    sim.add(body["horizon_id"])

sim.move_to_com()

# Finding the location of the Earth currently 
earth_index = None
for i, body in enumerate(body_list):
    if body["name"] == "Earth":
        earth_index = i
earth = sim.particles[earth_index]

# Probe settings
probe_start_offset = 0.0001
probe_launch_thrust = 0.5

# Earth's movement directions
earth_speed = math.sqrt(earth.vx**2 + earth.vy**2 + earth.vz**2) # Distance formula but for the velocity 
unit_vx = earth.vx / earth_speed
unit_vy = earth.vy / earth_speed
unit_vz = earth.vz / earth_speed

# Adding the probe
sim.add(
    m=1000/(1.989e30),
    # Placement of the probe, based on Earth
    x=earth.x + probe_start_offset,
    y=earth.y,
    z=earth.z,
    # Earth's starting velocity
    vx=earth.vx + probe_launch_thrust * unit_vx,
    vy=earth.vy + probe_launch_thrust * unit_vy,
    vz=earth.vz + probe_launch_thrust * unit_vz
)
probe = {
    "name": "Probe",
    "horizon_id": None,
    "size": 3,
    "colour": "purple"
}

body_list.append(probe)
probe_index = len(body_list) - 1


# 4. Setup Animation Parameters
years = 2
num_frames = 1000
times = [years * i / num_frames for i in range(num_frames)]

# Making a probe trajectory
probe_trajectory = []

# Remove Sun from plotting
plot_indices = []
plot_bodies = []

for i, body in enumerate(body_list):
    if body["name"] != "Sun":
        plot_indices.append(i)
        plot_bodies.append(body)

# Setup the figure and axes
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')
ax.set_xlabel('Distance (AU)')
ax.set_ylabel('Distance (AU)')
ax.set_title('Inner Solar System Animation')

# Create plot elements for each planet: (line for trail, dot for planet)
lines = [
    ax.plot(
        [],
        [],
        lw=1,
        color=body["colour"],
        label=body["name"]
    )[0]
    for body in plot_bodies
]

dots = [
    ax.plot(
        [],
        [],
        'o',
        color=body["colour"],
        markersize=body["size"]
    )[0]
    for body in plot_bodies
]


ax.legend(loc='upper right')

# Data storage for trails
x_data = [[] for _ in range(len(plot_bodies))]
y_data = [[] for _ in range(len(plot_bodies))]

# 4. Animation Function
def update(frame):
    t = times[frame]
    sim.integrate(t)
    
    for plot_i, particle_i in enumerate(plot_indices):
        p = sim.particles[particle_i]

        # Update trail data
        x_data[plot_i].append(p.x)
        y_data[plot_i].append(p.y)
            
        # Update the visual elements
        lines[plot_i].set_data(x_data[plot_i], y_data[plot_i])
        dots[plot_i].set_data([p.x], [p.y])

    probe_particle = sim.particles[probe_index]

    # New row of data for the trajectory
    probe_trajectory.append([
        frame,
            t,
            probe_particle.x,
            probe_particle.y,
            probe_particle.z,
            probe_particle.vx,
            probe_particle.vy,
            probe_particle.vz ])
        
    return lines + dots

# 5. Run and Save Animation
ani = FuncAnimation(fig, update, frames=num_frames, interval=30, blit=True)

# To save as MP4 (requires ffmpeg) or GIF (requires pillow)
# ani.save('solar_system_animation.mp4', writer='ffmpeg', fps=30)

print("Animation saved as solar_system_animation.mp4")
plt.show() # Uncomment to view while running

# Writing the entire trajectory into a csv file
with open("probe_trajectory.csv", "w", newline="") as file:
    csv_file = csv.writer(file)

    csv_file.writerow([
        "frame",
        "time_years",
        "x_AU",
        "y_AU",
        "z_AU",
        "vx_AU_per_year",
        "vy_AU_per_year",
        "vz_AU_per_year"
    ])

    csv_file.writerows(probe_trajectory)

print("Probe trajectory saved as probe_trajectory.csv")