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
        self.last_error = 0
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

class RK2:
    def __init__(self, f):
        self.f = f
        
    def step(self, t, x, u, h):
        k1 = self.f(t, x, u, h)
        k2 = self.f(t + 0.5 * h, x + 0.5 * h * k1, u, h)
        return x + h * k2

class RK4:       
    def __init__(self, f):
        self.f = f
        
    def step(self, t, x, u, h):
        k1 = self.f(t, x, u, h)
        k2 = self.f(t + 0.5 * h, x + 0.5 * h * k1, u, h)
        k3 = self.f(t + 0.5 * h, x + 0.5 * h * k2, u, h)
        k4 = self.f(t + h, x +  h * k3, u, h)
        return x + h/6*(k1 + 2*k2 + 2*k3 + k4)

class KalmanFilterND:
    def __init__(self, A, B, C, D, Q, R, x0=None, P0=None):
        self.A = np.atleast_2d(A)
        self.B = np.atleast_2d(B)
        self.C = np.atleast_2d(C)
        self.D = np.atleast_2d(D)

        self.n = self.A.shape[0]   # State vector dimesnion
        self.m = self.B.shape[1]   # Input vector dimension
        self.p = self.C.shape[0]   # Measurment dimension

        self.Q = np.atleast_2d(Q)
        self.R = np.atleast_2d(R)

        self.x = np.zeros((self.n, 1)) if x0 is None else np.atleast_2d(x0).reshape(-1, 1)
        self.P = np.eye(self.n) if P0 is None else np.atleast_2d(P0)
        
        self.gate = False
        self.gating_dist = 5.0
        
    def set_gate(self, use_gate, outlier_distance):
            self.gate = use_gate
            self.gating_dist = outlier_distance

    def predict(self, u=None):
        if u is None:
            u = np.zeros((self.m, 1))
        else:
            u = np.atleast_2d(u).reshape(-1, 1)

        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x
    
    def predict_dt(self, dt, q_c, u=None):
        if u is None:
            u = np.zeros((self.m, 1))
        else:
            u = np.atleast_2d(u).reshape(-1, 1)
    
        # === State transition matrix ===
        self.A = np.array([
            [1, dt, 0.5 * dt**2],
            [0,  1, dt],
            [0,  0, 1]
        ])
    
        # === Process noise covariance ===
        # Assume white noise on acceleration with spectral density q_c
        q_c = self.q_c  # You can store this constant in __init__
        F = self.A
        Q = np.array([
            [dt**5/20, dt**4/8, dt**3/6],
            [dt**4/8,  dt**3/3, dt**2/2],
            [dt**3/6,  dt**2/2, dt]
        ]) * q_c
        self.Q = Q
    
        # === Prediction ===
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    

    def update(self, z, u=None):
        if u is None:
            u = np.zeros((self.m, 1))
        else:
            u = np.atleast_2d(u).reshape(-1, 1)

        z = np.atleast_2d(z).reshape(-1, 1)

        y = z - (self.C @ self.x + self.D @ u)
        y = np.atleast_2d(y).reshape(-1, 1)
        
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        
        if self.gate:
            Mahalanobis = np.sqrt(y.T @ np.linalg.pinv(S) @ y)  
            if Mahalanobis > self.gating_dist:
                print("Measurement Gated. Returning prior.")
                return self.x

        self.x = self.x + K @ y
        I = np.eye(self.n)
        self.P = (I - K @ self.C) @ self.P
        return self.x
    
    def _update(self, z, H, R):
        z = np.atleast_2d(z).reshape(-1, 1)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        if self.gate:
            Mahalanobis = np.sqrt(y.T @ np.linalg.pinv(S) @ y)
            if Mahalanobis > self.gating_dist:
                print("Measurement Gated. Returning prior.")
                return self.x
        
        self.x = self.x + K @ y
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P
        return self.x

     
        
    def update_position(self, z):
        H = np.array([[1, 0, 0]])
        R = self.R_pos
        self._update(z, H, R)

    def update_acceleration(self, z):
        H = np.array([[0, 0, 1]])
        R = self.R_accel
        self._update(z, H, R)

        

class UKF:
    def __init__(self, n, m, f, h, Q, R, x0=None, P0=None, alpha=1e-3, beta=2, kappa=0):
        """
        Unscented Kalman Filter (UKF) using Van der Merwe's sigma points.

        Parameters
        ----------
        n : int
            State dimension
        m : int
            Measurement dimension
        f : callable
            Nonlinear process function: x_next = f(x, u, dt)
        h : callable
            Nonlinear measurement function: z = h(x)
        Q : ndarray (n,n)
            Process noise covariance
        R : ndarray (m,m)
            Measurement noise covariance
        x0 : ndarray (n,1)
            Initial state
        P0 : ndarray (n,n)
            Initial state covariance
        alpha, beta, kappa : float
            UKF tuning parameters
        """
        self.n = n
        self.m = m
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R

        self.x = np.zeros((n,1)) if x0 is None else x0.reshape(-1,1)
        self.P = np.eye(n) if P0 is None else P0

        # --- Compute scaling parameters for Van der Merwe sigma points ---
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (n + kappa) - n
        self.gamma = np.sqrt(n + self.lambda_)

        # --- Weights for mean and covariance ---
        self.Wm = np.full(2*n+1, 0.5/(n + self.lambda_))
        self.Wc = np.full(2*n+1, 0.5/(n + self.lambda_))
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc[0] = self.lambda_ / (n + self.lambda_) + (1 - alpha**2 + beta)

    def sigma_points(self, x, P):
        """Generate sigma points for the current mean and covariance"""
        n = self.n
        Psqrt = np.linalg.cholesky(P)
        sigma_pts = np.zeros((2*n + 1, n))
        sigma_pts[0] = x.ravel()
        for i in range(n):
            sigma_pts[i+1]   = x.ravel() + self.gamma * Psqrt[:, i]
            sigma_pts[n+i+1] = x.ravel() - self.gamma * Psqrt[:, i]
        return sigma_pts

    def predict(self, u=None, dt=0.01):
        """Prediction step of UKF"""
        sigma_pts = self.sigma_points(self.x, self.P)
        n_sigma = sigma_pts.shape[0]

        # Propagate through nonlinear dynamics
        X_pred = np.zeros_like(sigma_pts)
        for i in range(n_sigma):
            X_pred[i] = self.f(sigma_pts[i], u, dt).ravel()

        # Predicted mean
        self.x = np.sum(self.Wm[:,None] * X_pred, axis=0).reshape(-1,1)

        # Predicted covariance
        self.P = self.Q.copy()
        for i in range(n_sigma):
            dx = (X_pred[i] - self.x.ravel()).reshape(-1,1)
            self.P += self.Wc[i] * (dx @ dx.T)

        # Store predicted sigma points for update
        self.X_pred = X_pred

    def update(self, z):
        """Update step with measurement z"""
        sigma_pts = self.X_pred
        n_sigma = sigma_pts.shape[0]

        # Transform sigma points through measurement function
        Z_pred = np.zeros((n_sigma, self.m))
        for i in range(n_sigma):
            Z_pred[i] = self.h(sigma_pts[i]).ravel()

        # Predicted measurement mean
        z_pred = np.sum(self.Wm[:,None] * Z_pred, axis=0).reshape(-1,1)

        # Innovation covariance and cross-covariance
        S = self.R.copy()
        Cxz = np.zeros((self.n, self.m))
        for i in range(n_sigma):
            dz = (Z_pred[i] - z_pred.ravel()).reshape(-1,1)
            dx = (sigma_pts[i] - self.x.ravel()).reshape(-1,1)
            S += self.Wc[i] * dz @ dz.T
            Cxz += self.Wc[i] * dx @ dz.T

        # Kalman gain
        K = Cxz @ np.linalg.inv(S)

        # Update state and covariance
        y = z.reshape(-1,1) - z_pred
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        