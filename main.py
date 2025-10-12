import numpy as np
from simulation import Vehicle, Racetrack
from mpc_controller import MPCController
from gp_mpc_controller import GPMPCController
import matplotlib.pyplot as plt

def get_reference_state(vehicle, racetrack):
    waypoints = racetrack.waypoints
    distances = np.linalg.norm(waypoints - np.array([vehicle.x, vehicle.y]), axis=1)
    nearest_index = np.argmin(distances)

    # For a circular track, the next point is just the next index
    ref_index = (nearest_index + 1) % len(waypoints)
    ref_point = waypoints[ref_index]

    # Calculate reference yaw and velocity
    # This is a simplified approach
    ref_yaw = np.arctan2(ref_point[1] - vehicle.y, ref_point[0] - vehicle.x)
    ref_v = 1.5 # Constant reference velocity

    return np.array([ref_point[0], ref_point[1], ref_yaw, ref_v])

def run_simulation(controller, with_mismatch=False):
    vehicle = Vehicle(x=10.0, y=0.0, yaw=np.pi/2, v=0.0)
    racetrack = Racetrack()

    history_x = []
    history_y = []
    tracking_errors = []

    # GP-MPC specific
    gp_X_train = []
    gp_y_train = []

    for i in range(200):
        current_state = np.array([vehicle.x, vehicle.y, vehicle.yaw, vehicle.v])
        ref_state = get_reference_state(vehicle, racetrack)

        control_input = controller.solve(current_state, ref_state)
        a, delta = control_input[0], control_input[1]

        # Store previous state
        prev_state = np.copy(current_state)

        # Update vehicle
        vehicle.update(a, delta, 0.1)

        # Introduce model mismatch if enabled
        if with_mismatch:
            # e.g., an unmodeled disturbance or different vehicle parameter
            vehicle.x += 0.1 * np.cos(vehicle.yaw) # unmodeled wind
            vehicle.y += 0.1 * np.sin(vehicle.yaw)

        # Record data
        history_x.append(vehicle.x)
        history_y.append(vehicle.y)
        tracking_error = np.linalg.norm(current_state[:2] - ref_state[:2])
        tracking_errors.append(tracking_error)

        # GP-MPC: Online learning
        if isinstance(controller, GPMPCController):
            actual_next_state = np.array([vehicle.x, vehicle.y, vehicle.yaw, vehicle.v])
            predicted_next_state = prev_state + controller.f(prev_state, control_input) * controller.dt
            error = (actual_next_state - predicted_next_state.full().flatten()) / controller.dt

            gp_X_train.append(np.concatenate([prev_state, control_input]))
            gp_y_train.append(error)

            if i > 0 and i % 10 == 0: # Retrain every 10 steps
                controller.update_gp(np.array(gp_X_train), np.array(gp_y_train))

    return history_x, history_y, tracking_errors

if __name__ == '__main__':
    # Run with standard MPC
    mpc = MPCController()
    mpc_hist_x, mpc_hist_y, mpc_errors = run_simulation(mpc, with_mismatch=True)

    # Run with GP-MPC
    gp_mpc = GPMPCController()
    gp_mpc_hist_x, gp_mpc_hist_y, gp_mpc_errors = run_simulation(gp_mpc, with_mismatch=True)

    # Store results for visualization
    results = {
        'mpc': {'x': mpc_hist_x, 'y': mpc_hist_y, 'errors': mpc_errors},
        'gp_mpc': {'x': gp_mpc_hist_x, 'y': gp_mpc_hist_y, 'errors': gp_mpc_errors}
    }
    np.save('simulation_results.npy', results)
    print("Simulations complete. Results saved to simulation_results.npy")
