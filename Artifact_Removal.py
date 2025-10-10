import numpy as np
import matplotlib.pyplot as plt
import FilterLib as filt

from scipy import signal

# === PARAMETERS ===
dy = 0.012
y = np.arange(0, 0.36 + dy, dy)  # same y as before
f = 2.0               # sine wave frequency (cycles over y-domain)
noise_std = 0.05     # standard deviation of noise
outlier_magnitude = 1.0  # amplitude of outliers
outlier_every = 15   # every Nth sample is an outlier

# === SIGNAL GENERATION ===
clean_signal = np.sin(2 * np.pi * f * y / y[-1])  # sine wave across y
clean_signal = np.sin(2 * np.pi * f * y / y[-1])  # sine wave across y
noisy_signal = clean_signal + np.random.normal(0, noise_std, size=y.shape)

# Add periodic outliers
outlier_indices = np.arange(5, len(y), outlier_every)
noisy_signal[outlier_indices] += np.random.choice([-1, 1], size=len(outlier_indices)) * outlier_magnitude

# This is your z(y)
z = noisy_signal

# === PLOT z(y) ===
plt.figure(figsize=(10, 4))
plt.plot(y, z, 'o-', label='z(y) noisy sine with outliers')
plt.plot(y, clean_signal, '-', label='Clean sine signal', alpha=0.7)
plt.xlabel('y')
plt.ylabel('z(y)')
plt.title('Noisy sine wave with periodic outliers')
plt.grid(True)
plt.legend()
plt.show()

# === KALMAN FILTER SETUP ===
# Q = 1 * np.array([
#     [dy**5 / 20, dy**4 / 8,  dy**3 / 6],
#     [dy**4 / 8,  dy**3 / 3,  dy**2 / 2],
#     [dy**3 / 6,  dy**2 / 2,  dy      ]
# ])

# Q = 400 * np.array([
#     [dy**4/4, dy**3/2, dy**2/2],
#     [dy**3/2, dy**2,   dy],
#     [dy**2/2, dy,      1]
# ])

Q = 200 * np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1]
])

# Q = np.eye(3)*0.5
R = np.array([[0.00002]])      # measurement noise covariance
B = np.zeros((3, 1))        # no control input
D = np.zeros((1, 1))        # no direct feedthrough

C = np.array([[1, 0, 0]])   # measure position only

x0 = np.array([z[0],(z[1]-z[0])/dy,0])            # [x, xdot, xddot]
P0 = np.eye(3) * 1000.0

x_estimates = []

F = np.array([
    [1, dy, 0.5*dy**2],
    [0, 1, dy],
    [0, 0, 1]
])
kf = filt.KalmanFilterND(F, B, C, D, Q, R, x0=x0, P0=P0)
# kf.set_gate(True, 150)

for k in range(len(y)-1):
    dy = y[k+1] - y[k]

    F = np.array([
        [1, dy, 0.5*dy**2],
        [0, 1, dy],
        [0, 0, 1]
    ])
    kf.F = F
    prior = kf.predict()
    print(prior)
    kf.update(np.array([z[k]]))
    x_estimates.append(kf.x.copy())

x_estimates = np.array(x_estimates)

# === PLOT ORIGINAL AND FILTERED POSITION ===
plt.figure(figsize=(10, 4))
plt.plot(y[:-1], x_estimates[:, 0], 'r-', linewidth=2, label="Filtered position")
plt.plot(y, z, 'bo', markersize=4, label="Noisy measurements")
plt.plot(y, clean_signal, 'g--', linewidth=1.5, label="Clean sine signal")
plt.xlabel("y")
plt.ylabel("Position")
plt.title("Kalman Filter: Original vs Filtered Position")
plt.legend()
plt.grid(True)
plt.show()
