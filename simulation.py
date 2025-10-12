import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Vehicle:
    def __init__(self, x=0.0, y=-1.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = 2.5  # Wheelbase

    def update(self, a, delta, dt):
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / self.L * np.tan(delta) * dt
        self.v += a * dt

class Racetrack:
    def __init__(self, radius=10.0, num_points=100):
        self.radius = radius
        self.num_points = num_points
        self.waypoints = self.generate_waypoints()

    def generate_waypoints(self):
        theta = np.linspace(0, 2 * np.pi, self.num_points)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        return np.array([x, y]).T

class Simulation:
    def __init__(self):
        self.vehicle = Vehicle(x=10.0, y=0.0, yaw=np.pi/2, v=1.0)
        self.racetrack = Racetrack()
        self.dt = 0.1
        self.fig, self.ax = plt.subplots()
        self.history_x = []
        self.history_y = []

    def animate(self, i):
        # Dummy control inputs for now
        a = 0.1
        delta = 0.05
        self.vehicle.update(a, delta, self.dt)

        self.history_x.append(self.vehicle.x)
        self.history_y.append(self.vehicle.y)

        self.ax.clear()
        self.ax.plot(self.racetrack.waypoints[:, 0], self.racetrack.waypoints[:, 1], 'b--')
        self.ax.plot(self.history_x, self.history_y, 'r-')
        self.ax.plot(self.vehicle.x, self.vehicle.y, 'ro')
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-15, 15)
        self.ax.set_title("Vehicle Simulation")

    def run(self):
        ani = FuncAnimation(self.fig, self.animate, frames=200, interval=100)
        plt.show()

if __name__ == '__main__':
    sim = Simulation()
    sim.run()
