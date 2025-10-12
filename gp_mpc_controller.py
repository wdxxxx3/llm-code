import casadi as ca
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GPMPCController:
    def __init__(self, N=10, dt=0.1):
        self.N = N
        self.dt = dt
        self.L = 2.5

        # Gaussian Process Model
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        self.gp_trained = False

        # Casadi symbolic variables and functions
        x, y, yaw, v = ca.SX.sym('x'), ca.SX.sym('y'), ca.SX.sym('yaw'), ca.SX.sym('v')
        states = ca.vertcat(x, y, yaw, v)
        n_states = states.numel()

        a, delta = ca.SX.sym('a'), ca.SX.sym('delta')
        controls = ca.vertcat(a, delta)
        n_controls = controls.numel()

        # Vehicle model
        rhs = ca.vertcat(v * ca.cos(yaw), v * ca.sin(yaw), v / self.L * ca.tan(delta), a)
        self.f = ca.Function('f', [states, controls], [rhs])

        # Define the optimization problem once
        self.opti = ca.Opti()
        self.U = self.opti.variable(n_controls, self.N)
        self.X = self.opti.variable(n_states, self.N + 1)
        self.P = self.opti.parameter(n_states + n_states) # Initial state and reference state for the horizon
        self.GP_C = self.opti.parameter(n_states, self.N) # GP correction term for the horizon

        # Cost function
        Q = np.diag([1.0, 1.0, 0.5, 0.5])
        R = np.diag([0.1, 0.5])
        cost = 0
        for k in range(self.N):
            # For simplicity, we'll use a single reference state for the entire horizon
            ref_state = self.P[n_states:]
            cost += ca.mtimes([(self.X[:, k] - ref_state).T, Q, (self.X[:, k] - ref_state)])
            cost += ca.mtimes([self.U[:, k].T, R, self.U[:, k]])
        self.opti.minimize(cost)

        # Constraints
        self.opti.subject_to(self.X[:, 0] == self.P[:n_states])
        for k in range(self.N):
            x_next = self.X[:, k] + self.f(self.X[:, k], self.U[:, k]) * self.dt + self.GP_C[:, k] * self.dt
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # Control input constraints
        self.opti.subject_to(self.opti.bounded(-1.0, self.U[0, :], 1.0))
        self.opti.subject_to(self.opti.bounded(-0.5, self.U[1, :], 0.5))

        # Solver
        opts = {'ipopt.print_level': 0, 'print_time': 0}
        self.opti.solver('ipopt', opts)

    def update_gp(self, X_train, y_train):
        if len(X_train) > 0:
            self.gp.fit(X_train, y_train)
            self.gp_trained = True

    def solve(self, current_state, ref_state):
        # Set the initial state and reference state parameter
        self.opti.set_value(self.P, np.concatenate((current_state, ref_state)))

        # Predict the GP correction for the entire horizon
        gp_corrections_horizon = np.zeros((4, self.N))
        if self.gp_trained:
            # For simplicity, we assume the GP correction is constant over the horizon,
            # predicted based on the current state and a zero control input.
            # A more advanced approach would predict the correction at each step of the horizon.
            X_pred = np.concatenate([current_state, np.zeros(2)]) # Current state and zero control
            gp_correction, _ = self.gp.predict(X_pred.reshape(1, -1), return_std=True)
            gp_corrections_horizon = np.tile(gp_correction.T, (1, self.N))

        # Set the GP correction parameter
        self.opti.set_value(self.GP_C, gp_corrections_horizon)

        # Solve the optimization problem
        sol = self.opti.solve()
        return sol.value(self.U[:, 0])

if __name__ == '__main__':
    controller = GPMPCController()
    current_state = np.array([0.0, 0.0, 0.0, 0.0])
    ref_state = np.array([1.0, 1.0, 0.0, 1.0])

    # Dummy training data
    X_train = np.random.rand(10, 6) # 10 samples, 4 states + 2 controls
    y_train = np.random.rand(10, 4) # 10 samples, 4 state errors
    controller.update_gp(X_train, y_train)

    control_input = controller.solve(current_state, ref_state)
    print("Optimal control input (with GP):", control_input)
