import numpy as np
import matplotlib.pyplot as plt
import FilterLib as filt

# Time parameters
dt = 0.01          # time step
T = 2.5            # total time (s)
t = np.arange(0, T, dt)

# Create Filters
exp = filt.ExponentialFilter(0.496, 0)
gh = filt.GHFilter(0.496, 0.084, 0, 0, dt)
ghk = filt.GHKFilter(0.611, 0.189, 0.005, 0, 1, 0, dt)

# Initialize signal
x = np.zeros_like(t)

# Define segment times
t_const1_end = 0.3
t_ramp_end = 0.7
t_const2_end = 1.0
t_quad_end = 2.0  # longer quadratic ramp

# Segment values
x[t <= t_const1_end] = 0.5                       # initial constant
x[(t > t_const1_end) & (t <= t_ramp_end)] = np.linspace(0.5, 2.0, sum((t > t_const1_end) & (t <= t_ramp_end)))  # linear ramp
x[(t > t_ramp_end) & (t <= t_const2_end)] = 2.0  # second constant

# Aggressive quadratic ramp: larger coefficient
quad_t = t[(t > t_const2_end) & (t <= t_quad_end)]
quad_coeff = 2.5  # increase to make ramp more aggressive
x[(t > t_const2_end) & (t <= t_quad_end)] = 2.0 + quad_coeff * (quad_t - t_const2_end)**2

x[t > t_quad_end] = 4.5                        # final constant after quadratic ramp

# Initialize arrays for filtering
y = np.zeros_like(t)
y_exp = np.zeros_like(t)
y_gh = np.zeros_like(t)
y_ghk = np.zeros_like(t)

# Noise
noise_std = 0.0

# Simulate measurement and filtering
for i in range(len(t)):
    y[i] = x[i] + np.random.normal(0, noise_std)
    y_exp[i] = exp.update(y[i])
    y_gh[i] = gh.update(y[i])
    y_ghk[i] = ghk.update(y[i])

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t, x, label='True Output (x)', linewidth=2)
# plt.plot(t, y, label='Noisy Measurement (y)', linestyle='--', alpha=0.7)
plt.plot(t, y_exp, label='Filtered Output (y_exp)', linewidth=2)
plt.plot(t, y_gh, label='Filtered Output (y_gh)', linewidth=2)
# plt.plot(t, y_ghk, label='Filtered Output (y_ghk)', linewidth=2)
plt.xlabel('Time [s]')
plt.ylabel('Output')
plt.title('Piecewise Reference')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
