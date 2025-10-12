import casadi as ca
import numpy as np

class MPCController:
    def __init__(self, N=10, dt=0.1):
        self.N = N  # Prediction horizon
        self.dt = dt
        self.L = 2.5  # Wheelbase

        # Define the symbolic variables
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        v = ca.SX.sym('v')
        states = ca.vertcat(x, y, yaw, v)
        n_states = states.numel()

        a = ca.SX.sym('a')
        delta = ca.SX.sym('delta')
        controls = ca.vertcat(a, delta)
        n_controls = controls.numel()

        # Define the vehicle model
        rhs = ca.vertcat(
            v * ca.cos(yaw),
            v * ca.sin(yaw),
            v / self.L * ca.tan(delta),
            a
        )
        self.f = ca.Function('f', [states, controls], [rhs])

        # Define the optimization problem
        self.opti = ca.Opti()
        self.U = self.opti.variable(n_controls, self.N)
        self.X = self.opti.variable(n_states, self.N + 1)
        self.P = self.opti.parameter(n_states + n_states) # Initial state and reference state

        # Cost function
        Q = np.diag([1.0, 1.0, 0.5, 0.5])
        R = np.diag([0.1, 0.5])
        cost = 0
        for k in range(self.N):
            cost += ca.mtimes([(self.X[:, k] - self.P[n_states:]).T, Q, (self.X[:, k] - self.P[n_states:])])
            cost += ca.mtimes([self.U[:, k].T, R, self.U[:, k]])
        self.opti.minimize(cost)

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.P[:n_states])
        for k in range(self.N):
            x_next = self.X[:, k] + self.f(self.X[:, k], self.U[:, k]) * self.dt
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # Control input constraints
        self.opti.subject_to(self.opti.bounded(-1.0, self.U[0, :], 1.0))
        self.opti.subject_to(self.opti.bounded(-0.5, self.U[1, :], 0.5))

        # Solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.opti.solver('ipopt', opts)

    def solve(self, current_state, ref_state):
        self.opti.set_value(self.P, np.concatenate((current_state, ref_state)))
        sol = self.opti.solve()
        return sol.value(self.U[:, 0])

if __name__ == '__main__':
    # Example usage
    controller = MPCController()
    current_state = np.array([0.0, 0.0, 0.0, 0.0])
    ref_state = np.array([1.0, 1.0, 0.0, 1.0])
    control_input = controller.solve(current_state, ref_state)
    print("Optimal control input:", control_input)
