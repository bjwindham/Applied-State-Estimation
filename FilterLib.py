import numpy as np
import matplotlib.pyplot as plt 

class ExponentialFilter:
    def __init__(self, alpha: float, initial_value: float = 0.0):
        if not (0 < alpha <= 1):
            raise ValueError("alpha must be in the range (0, 1]")
        
        self.alpha = alpha
        self.value = initial_value

    def update(self, measurement: float) -> float:
        self.value = (1 - self.alpha) * self.value + self.alpha * measurement
        return self.value


    
class GHFilter:
    def __init__(self, g, h, x0, x_dot0, dt):
        self.g = g
        self.h = h
        self.x_dot = x_dot0
        self.dt = dt
        self.x = x0
        self.x_pred = x0
        self.x_dot_pred = x_dot0

    def update(self, measurement):
        # Prediction step
        self.x_pred = self.x + self.x_dot * self.dt
        self.x_dot_pred = self.x_dot

        # Residual
        residual = measurement - self.x_pred

        # Update step
        self.x = self.x_pred + self.g * residual
        self.x_dot = self.x_dot + self.h * residual / self.dt

        return self.x

        
        
class GHKFilter:
    def __init__(self, g, h, k, x0, x_dot0, x_ddot0, dt):
        self.g = g
        self.h = h
        self.k = k
        self.dt = dt
        self.x = x0
        self.x_dot = x_dot0
        self.x_ddot = x_ddot0
        self.x_pred = x0
        self.x_dot_pred = x_dot0
        self.x_ddot_pred = x_ddot0

    def update(self, measurement):
        # Predict
        self.x_pred = self.x + self.x_dot * self.dt + 0.5 * self.x_ddot * self.dt**2
        self.x_dot_pred = self.x_dot + self.x_ddot * self.dt
        self.x_ddot_pred = self.x_ddot

        # Residual
        r = measurement - self.x_pred

        # Update
        self.x = self.x_pred + self.g * r
        self.x_dot = self.x_dot_pred + self.h * r / self.dt 
        self.x_ddot = self.x_ddot_pred + (2 * self.k / self.dt**2) * r

        return self.x

        
class Kalman1D:
    def __init__(self, variance, x, process_variance, measurement_variance, A, B):
        self.P = variance
        self.x = x
        self.Q = process_variance
        self.R = measurement_variance
        self.A = A
        self.B = B
        
    def predict(self, u):
        self.x = self.A*self.x + self.B*u  
        self.P = self.P + self.Q
        
    def update(self, z):
        y = z - self.x
        K = self.P/(self.P + self.R)
        self.x = self.x + K*y
        self.P = (1-K)*self.P
        
class PID:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.err_sum = 0
        self.last_state = 0
        self.last_state_2nd = 0
        self.dt = dt
        
    def compute_control(self, state, reference):
        error = reference - state
        self.err_sum = self.err_sum + error*self.dt
        
        P = self.kp*error
        I = self.err_sum*self.ki
        D = ((3*state - 4*self.last_state + self.last_state_2nd)/(2*self.dt))*self.kd
        u = P + I - D
        
        self.last_state_2nd = self.last_state
        self.last_state = state
  
        return u
        
class ARX:

    def __init__(self, n_a, n_b, delay=1):
        self.n_a = n_a 
        self.n_b = n_b
        self.delay = delay
        self.a = None
        self.b = None
        self.residuals = None
        self.A = None
        self.B = None
        self.C = None
        self.Q = None
    
    def fit(self, y, u):
        N = len(y)
        Phi = []
        Y = []
        for k in range(max(self.n_a, self.n_b + self.delay - 1), N):
            phi_row = []
            # past outputs (with minus sign for ARX convention)
            for i in range(1, self.n_a+1):
                phi_row.append(-y[k-i])
            # past inputs
            for j in range(self.n_b):
                phi_row.append(u[k-j-self.delay+1])
            Phi.append(phi_row)
            Y.append(y[k])
        Phi = np.array(Phi)
        Y = np.array(Y)
        
        # Solve least squares
        theta, residuals, rank, s = np.linalg.lstsq(Phi, Y, rcond=None)
        self.a = theta[:self.n_a]
        self.b = theta[self.n_a:]
        self.residuals = residuals
        
        return self.a, self.b, self.residuals

import numpy as np

class StateSpaceSimulator:
    def __init__(self, A, B, C, D, x0=None, Q=None, R=None):
        """
        Discrete-time state-space simulator:
            x[k+1] = A x[k] + B u[k] + w[k]
            y[k]   = C x[k] + D u[k] + v[k]

        Parameters
        ----------
        A, B, C, D : np.ndarray
            System matrices.
        x0 : np.ndarray, optional
            Initial state vector (defaults to zeros).
        Q : np.ndarray or float, optional
            Process noise covariance (default None = no process noise).
        R : np.ndarray or float, optional
            Measurement noise covariance (default None = no measurement noise).
        """
        self.A = np.atleast_2d(A)
        self.B = np.atleast_2d(B)
        self.C = np.atleast_2d(C)
        self.D = np.atleast_2d(D)
        self.n = self.A.shape[0]  # state dimension
        self.m = self.B.shape[1]  # input dimension
        self.p = self.C.shape[0]  # output dimension

        # initial condition
        self.x = np.zeros((self.n, 1)) if x0 is None else np.atleast_2d(x0).reshape(-1, 1)

        # noise covariances
        self.Q = Q if Q is not None else 0.0
        self.R = R if R is not None else 0.0

    def step(self, u):
        """
        Advance system by one step.
        Parameters
        ----------
        u : array-like
            Input vector at this timestep.
        Returns
        -------
        y : np.ndarray
            Output vector at this timestep.
        """
        u = np.atleast_2d(u).reshape(-1, 1)

        # Process noise
        w = np.random.multivariate_normal(np.zeros(self.n), 
                                          self.Q if np.ndim(self.Q) == 2 else np.eye(self.n)*self.Q).reshape(-1, 1) \
            if np.any(self.Q) else 0

        # Measurement noise
        v = np.random.multivariate_normal(np.zeros(self.p), 
                                          self.R if np.ndim(self.R) == 2 else np.eye(self.p)*self.R).reshape(-1, 1) \
            if np.any(self.R) else 0

        # State update
        self.x = self.A @ self.x + self.B @ u + w

        # Output
        y = self.C @ self.x + self.D @ u + v
        return y


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        