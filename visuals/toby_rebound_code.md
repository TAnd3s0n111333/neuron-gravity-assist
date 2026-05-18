import rebound
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 1. Initialize the simulation
sim = rebound.Simulation()
sim.units = ('yr', 'AU', 'Msun')

# 2. Add the Sun and Planets
planets = ["Sun", "Mercury", "Venus", "Earth", "Mars"]
for planet in planets:
    sim.add(planet)

sim.move_to_com()

# 3. Setup Animation Parameters
years = 2
num_frames = 200
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
dots = [ax.plot([], [], 'o')[0] for i in range(len(planets))]
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
# plt.show() # Uncomment to view while running