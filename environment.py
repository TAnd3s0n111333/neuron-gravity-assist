# pyrefly: ignore [missing-import]
import rebound
import numpy as np
import matplotlib.pyplot as plt

sim = rebound.Simulation()

# The integrator chosen because it is better for predicting with collisions
sim.integrator = "ias15" #Good for collisions but not for speed

# Using NASA JPL Horizon
sim.add("399") # Earth
sim.add("301") # Moon

sim.move_to_com() # Shifts the centre of mass #To where? 

# Amount of photos & days to be confirmed
photos = 100
days = 365
x_pos = np.empty((2,photos))
y_pos = np.empty((2,photos))
times = np.linspace(0, days, photos)

print(x_pos)

for i, t in enumerate(times):
    if i%10==0:
        print(x_pos)
    sim.integrate(t)
    # print(len(x_pos[0]))
    # Saving the position of Earth
    x_pos[0, i] = sim.particles[0].x
    y_pos[0, i] = sim.particles[0].y
    # Saving the position of Moon
    x_pos[1, i] = sim.particles[1].x
    y_pos[1, i] = sim.particles[1].y

print(len(x_pos))
print(len(y_pos))

# Plotting it
plt.scatter(x_pos[0], y_pos[0], s=8,  c='blue', label='Earth')
plt.scatter(x_pos[1], y_pos[1], s=3,  c='gray', label='Moon')
plt.legend()
plt.show()

#Update function
# def update(i):
#     plt.scatter(x_pos[0, i], y_pos[0, i], s=8,  c='blue', label='Earth')
#     plt.scatter(x_pos[1, i], y_pos[1, i], s=3,  c='gray', label='Moon')
#     plt.legend()
#     plt.show()