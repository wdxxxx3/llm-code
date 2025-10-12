import numpy as np
import matplotlib.pyplot as plt
from simulation import Racetrack

def visualize_results():
    results = np.load('simulation_results.npy', allow_pickle=True).item()
    racetrack = Racetrack()

    mpc_results = results['mpc']
    gp_mpc_results = results['gp_mpc']

    # Plot trajectories
    plt.figure(figsize=(10, 10))
    plt.plot(racetrack.waypoints[:, 0], racetrack.waypoints[:, 1], 'b--', label='Racetrack')
    plt.plot(mpc_results['x'], mpc_results['y'], 'r-', label='Standard MPC')
    plt.plot(gp_mpc_results['x'], gp_mpc_results['y'], 'g-', label='GP-MPC')
    plt.title('Vehicle Trajectories')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('trajectories.png')
    print("Trajectories plot saved to trajectories.png")

    # Plot tracking errors
    plt.figure(figsize=(10, 5))
    plt.plot(mpc_results['errors'], 'r-', label='Standard MPC')
    plt.plot(gp_mpc_results['errors'], 'g-', label='GP-MPC')
    plt.title('Tracking Error Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Tracking Error (m)')
    plt.legend()
    plt.grid(True)
    plt.savefig('tracking_errors.png')
    print("Tracking errors plot saved to tracking_errors.png")

if __name__ == '__main__':
    visualize_results()
