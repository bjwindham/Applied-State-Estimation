import numpy as np
import matplotlib.pyplot as plt
from FilterLib import KalmanFilterND, StateSpaceSimulator

# True signal /  Measurements
dt = 0.1
T = 15
N = int(T / dt)

t = np.linspace(0, T, N)

# cubic polynomial trajectories (constant acceleration)
a_x, b_x, c_x, d_x = 0.001, -0.02, 0.5, 0.0
a_y, b_y, c_y, d_y = -0.0015, 0.03, 0.1, 0.0

x_true = a_x * t**3 + b_x * t**2 + c_x * t + d_x
y_true = a_y * t**3 + b_y * t**2 + c_y * t + d_y

stdev = 0.05
x_meas = x_true + np.random.normal(0, stdev, size=N)
y_meas = y_true + np.random.normal(0, stdev, size=N)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(x_true, y_true, label="True trajectory", linewidth=2)
plt.scatter(x_meas, y_meas, s=15, color="red", alpha=0.6, label="Measured trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.title("True vs Measured Trajectory")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()


# State transition F
F = np.array([
    [1, 0,    dt, 0,  0.5*dt**2, 0],
    [0, 1,    0,  dt, 0,         0.5*dt**2],
    [0, 0,    1,  0,  dt,        0],
    [0, 0,    0,  1,  0,         dt],
    [0, 0,    0,  0,  1,         0],
    [0, 0,    0,  0,  0,         1]
], dtype=float)

# Measurement matrix H:
H = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0]
], dtype=float)

# Measurement covariance R
R = 0.2 * np.eye(2)

# Process noise
phi = 0.02
Q = np.array([
    [ (1/20) * dt**5,     0,                (1/8) * dt**4,     0,                (1/6) * dt**3, 0              ],
    [ 0,                  (1/20) * dt**5,   0,                 (1/8) * dt**4,   0,              (1/6) * dt**3 ],
    [ (1/8) * dt**4,      0,                (1/3) * dt**3,     0,                (1/2) * dt**2, 0              ],
    [ 0,                  (1/8) * dt**4,    0,                 (1/3) * dt**3,   0,              (1/2) * dt**2 ],
    [ (1/6) * dt**3,      0,                (1/2) * dt**2,     0,                dt,             0              ],
    [ 0,                  (1/6) * dt**3,    0,                 (1/2) * dt**2,   0,              dt             ]
], dtype=float) * phi

# Initial state and covariance
x0 = np.zeros(6)
P0 = np.eye(6) * 100.0

# No control input (kinematic Kalman filter) and no D matrix
B = np.zeros((6, 1))
D = np.zeros((2, 1))

# Instantiate filter
kf = KalmanFilterND(F, B, H, D, Q, R, x0=x0, P0=P0)
filtered_vals = []

for i in range(N):

    kf.predict()
    x_est = kf.update(np.array([x_meas[i], y_meas[i]]))
    filtered_vals.append(x_est.copy())


filtered_vals = np.array(filtered_vals)
x_filt = filtered_vals[:, 0]
y_filt = filtered_vals[:, 1]

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(x_true, y_true, label="True trajectory", linewidth=2)
plt.scatter(x_meas, y_meas, s=15, color="red", alpha=0.4, label="Measured")
plt.plot(x_filt, y_filt, color="green", linewidth=2, label="Kalman estimate")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Kalman-Filtered Trajectory")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
    
    

    
    
    
    