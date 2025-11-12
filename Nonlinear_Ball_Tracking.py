import numpy as np
import matplotlib.pyplot as plt
import FilterLib as filt


def ball_dynamics(t, x, u, h):

    g = 9.81      # m/s^2
    m = 1.0       # kg
    c = 0.1       # drag coefficient 

    vx, vy = x[2], x[3]
    v = np.hypot(vx, vy)

    ax = -(c / m) * v * vx
    ay = -g - (c / m) * v * vy

    return np.array([vx, vy, ax, ay])


# Simulate the ball in the air with RK4
rk4 = filt.RK4(ball_dynamics)

# Initial state: [x, y, x_dot, y_dot]
x = np.array([0.0, 0.0, 30.0, 30.0])
u = None
h = 0.01
t = 0.0

trajectory = [x.copy()]
times = [t]

# Run until the ball hits the ground
while x[1] >= 0.0:
    x = rk4.step(t, x, u, h)
    t += h
    trajectory.append(x.copy())
    times.append(t)

trajectory = np.array(trajectory)
times = np.array(times)


# Add Gaussian noise
noise_std = 0.2  # meters
noisy_measurements = trajectory[:, :2] + np.random.normal(0, noise_std, trajectory[:, :2].shape)


#%% Linear tracking Kalman Filter
g = 9.81  # m/s^2

A = np.array([[1, 0, h, 0],
              [0, 1, 0, h],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

B = np.array([[0.0],
              [0.5 * h**2],
              [0.0],
              [h]])

C = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

D = np.zeros((2, 1))

# Covariances
Q = np.eye(4) * 0.002
R = np.eye(2) * 0.2

# Initial state and covariance
x0 = np.array([[0.0], [0.0], [30.0], [30.0]])
P0 = np.eye(4)

# Instantiate Kalman filter
kf = filt.KalmanFilterND(A, B, C, D, Q, R, x0=x0, P0=P0)

# Gravity input vector for each step
u = np.array([[g]])


# Preallocate array for KF estimates
N = noisy_measurements.shape[0]
x_estimates = np.zeros((N, 4))  # store [x, y, vx, vy] at each step

# Gravity input
u = np.array([[g]])

# Loop over measurements
for k in range(N):
    z = noisy_measurements[k]  # current measurement [x, y]
    
    # --- Prediction step ---
    kf.predict(u=u)
    
    # --- Update step ---
    kf.update(z)
    
    # --- Store current estimate ---
    x_estimates[k, :] = kf.x.ravel()
    
#%% UKF Implementation

# UKF process function using your existing RK4 dynamics
def f_nl(x, u, dt):
    return rk4.step(0, x, u, dt)

# UKF measurement function: position only
def h_pos(x):
    return x[:2]

n = 4  # state: [x, y, vx, vy]
m = 2  # measurement: [x, y]

ukf = filt.UKF(
    n=n,
    m=m,
    f=f_nl,
    h=h_pos,
    Q=np.eye(n)*0.002,  # tune similarly to KF
    R=np.eye(m)*0.2,
    x0=x0,
    P0=P0
)

ukf_estimates = np.zeros((N, n))

for k in range(N):
    ukf.predict(u=np.array([[g]]), dt=h)
    ukf.update(noisy_measurements[k])
    ukf_estimates[k, :] = ukf.x.ravel()


#%% Plot the ball's trajectory
plt.figure(figsize=(6, 4))

# True trajectory
plt.plot(trajectory[:, 0], trajectory[:, 1], label="True trajectory", lw=2)

# Noisy measurements
plt.scatter(noisy_measurements[:, 0], noisy_measurements[:, 1], 
            s=10, alpha=0.6, label="Noisy measurements")

# Kalman Filter estimates
plt.plot(x_estimates[:, 0], x_estimates[:, 1], color='g', lw=2, label="KF estimate")

# UKF estimates
plt.plot(ukf_estimates[:, 0], ukf_estimates[:, 1], color='r', lw=2, label="UKF estimate")

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("2D Projectile Tracking: True vs Noisy vs KF vs UKF")
plt.grid(True)
plt.axis("equal")   # keep physical scaling consistent
plt.legend()
plt.show()

