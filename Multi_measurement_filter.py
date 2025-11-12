import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import FilterLib as filt
from scipy.signal import butter, filtfilt

import numpy as np

def forward_backward_exp_filter(z, alpha):

    z = np.asarray(z)
    N = len(z)
    y_forward = np.empty_like(z)
    y_backward = np.empty_like(z)

    # --- Forward pass ---
    y_forward[0] = z[0]
    for i in range(1, N):
        y_forward[i] = alpha * z[i] + (1 - alpha) * y_forward[i-1]

    # --- Backward pass ---
    y_backward[-1] = y_forward[-1]
    for i in range(N-2, -1, -1):
        y_backward[i] = alpha * y_forward[i] + (1 - alpha) * y_backward[i+1]

    return y_backward


def zero_phase_highpass(data, fs, cutoff, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)  # forward-backward zero-phase
    return filtered_data

camera_csv = "phone_positions_1hz.csv"
accel_csv  = "Accel_1Hz.csv"

# Read camera positions
cam_df = pd.read_csv(camera_csv)
cam_df.columns = cam_df.columns.str.strip().str.replace(' ', '_')
cam_time = cam_df['time'].to_numpy()
cam_x    = cam_df['x_m'].to_numpy()
cam_x = cam_x-cam_x[0]

# Normalize camera time to start at 0
cam_time = cam_time - cam_time[0]

# accelerometer data
accel_df = pd.read_csv(accel_csv, comment='#') 
accel_df.columns = accel_df.columns.str.strip().str.replace(' ', '_')
accel_time = accel_df['time'].to_numpy()
accel_x    = accel_df['ax'].to_numpy()

# accel_x = forward_backward_exp_filter(accel_x, 0.1)
fs = 100.0      # sampling frequency (Hz)
cutoff = 0.05   # high-pass cutoff to remove drift (~0.05 Hz)
# accel_x = zero_phase_highpass(accel_x, fs, cutoff)


accel_time = accel_time - accel_time[0]

# Stack sensor data
cam_array = np.column_stack((cam_time, cam_x, np.zeros_like(cam_time)))
accel_array = np.column_stack((accel_time, accel_x, np.ones_like(accel_time)))

# Combine and sort by time
combined_array = np.vstack((cam_array, accel_array))
combined_array = combined_array[np.argsort(combined_array[:, 0])]

# Kalman Filter Implementation
kf = filt.KalmanFilterND(A=np.eye(3), B=np.zeros((3,1)), C=np.eye(3), D=np.zeros((1,1)),
                    Q=np.eye(3), R=np.eye(3), x0=np.zeros(3), P0=np.eye(3))

# Set sensor noise
kf.R_pos = np.array([[1e-4]])   # extremely low → trust position
kf.R_accel = np.array([[1]])  # very high → ignore accel
kf.q_c = 1         # small process noise → smooth prediction
                    # Process noise spectral density

last_time = combined_array[0,0]
estimates = []

for row in combined_array:
    t, value, label = row
    dt = t - last_time
    last_time = t

    # Variable update rate predict
    kf.predict_dt(dt, q_c=kf.q_c)

    # Update based on sensor type
    if label == 0:
        kf.update_position(value)
    else:
        kf.update_acceleration(value)

    estimates.append(kf.x.flatten())

estimates = np.array(estimates)

time = combined_array[:, 0]

# Plot
plt.figure(figsize=(10, 8))

# Position
plt.subplot(3, 1, 1)
plt.plot(time, estimates[:, 0], label='Estimated Position', color='b')
plt.scatter(cam_time, cam_x, color='r', s=15, label='Camera Position')
plt.ylabel('Position (m)')
plt.title('Kalman Filter Estimates')
plt.legend()
plt.grid(True)

# Velocity
plt.subplot(3, 1, 2)
plt.plot(time, estimates[:, 1], label='Estimated Velocity', color='g')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid(True)

# Acceleration
plt.subplot(3, 1, 3)
plt.plot(time, estimates[:, 2], label='Estimated Acceleration', color='m')
plt.scatter(accel_time, accel_x, color='c', s=10, alpha=0.5, label='Measured Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()