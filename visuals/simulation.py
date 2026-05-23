import rebound
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
import csv

class Simulation:
    def __init__(self):
        # 1. Initialize the simulation
        self.initial_sim = rebound.Simulation()
        self.initial_sim.units = ('yr', 'AU', 'Msun')
        self.initial_sim.integrator = "whfast"

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

        self.body_list = list(solar_system.values())
        for body in self.body_list:
            self.initial_sim.add(body["horizon_id"])

        self.initial_sim.move_to_com()

        # Finding the location of the Earth currently 
        earth_index = None
        for i, body in enumerate(self.body_list):
            if body["name"] == "Earth":
                earth_index = i

        earth = self.initial_sim.particles[earth_index]

        # Probe settings
        probe_start_offset = 0.0001
        probe_launch_thrust = 0.5

        # Earth's movement directions
        earth_speed = math.sqrt(earth.vx**2 + earth.vy**2 + earth.vz**2) # Distance formula but for the velocity 
        unit_vx = earth.vx / earth_speed
        unit_vy = earth.vy / earth_speed
        unit_vz = earth.vz / earth_speed

        # Adding the probe
        self.initial_sim.add(
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

        self.body_list.append(probe)
        self.probe_index = len(self.body_list) - 1

    def _get_future_trajectory(self, steps=200, dt=0.01):
        future_sim = self.initial_sim.copy()
        xs, ys = [], []
        for _ in range(steps):
            sc = future_sim.particles[self.probe_index]
            xs.append(sc.x)
            ys.append(sc.y)
            future_sim.integrate(future_sim.t + dt)
        return xs, ys

    def run_inference(self):
        sim = self.initial_sim.copy()

        # 4. Setup Animation Parameters
        years = 2
        num_frames = 1000
        times = [years * i / num_frames for i in range(num_frames)]

        # Making a probe trajectory
        probe_trajectory = []

        # Remove Sun from plotting
        plot_indices = []
        plot_bodies = []

        for i, body in enumerate(self.body_list):
            if body["name"] != "Sun":
                plot_indices.append(i)
                plot_bodies.append(body)

        # Setup the figure and axes
        view_radius = 2.0  # AU — half-width of the camera window around the probe

        fig, ax = plt.subplots(figsize=(8, 8))
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

        future_line, = ax.plot([], [], ':', color='purple', lw=1, alpha=0.6, label='Predicted path')

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

            probe_particle = sim.particles[self.probe_index]

            # Camera tracks the probe
            ax.set_xlim(probe_particle.x - view_radius, probe_particle.x + view_radius)
            ax.set_ylim(probe_particle.y - view_radius, probe_particle.y + view_radius)

            # Plot predicted coasting trajectory
            fx, fy = self._get_future_trajectory()
            future_line.set_data(fx, fy)

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
                
            return lines + dots + [future_line]

        # 5. Run and Save Animation
        ani = FuncAnimation(fig, update, frames=num_frames, interval=30, blit=False)

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

sim = Simulation()
sim.run_inference()