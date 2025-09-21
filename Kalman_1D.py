import numpy as np
import matplotlib.pyplot as plt
import FilterLib as fl  

tau = 50.0       
K = 25.0          
dt = 1.0         

# Discretized system coefficients
A = np.array([[np.exp(-dt/tau)]])
B = np.array([[K * (1 - np.exp(-dt/tau))]])
C = np.array([[1.0]])
D = np.array([[0.0]])

# Noise 
Q = 0.1     
R = 0.1  

sys = fl.StateSpaceSimulator(A, B, C, D, Q=Q, R=R, x0=[0])
sys_ideal = fl.StateSpaceSimulator(A, B, C, D, Q=0, R=0, x0=[0])

# Set up sim and inputs
N = 400
u = np.zeros(N)
u[20:280] = 1.0
u[280:] = 0.0

y = []
y_ideal = []
for k in range(N):
    yk = sys.step(u[k])
    yk_ideal = sys_ideal.step(u[k])
    y.append(float(yk))
    y_ideal.append(float(yk_ideal))

# Plot
time = np.arange(N) * dt
plt.figure()
plt.plot(time, y, label="Measured Output")
plt.plot(time, y_ideal, label="Ideal Output")
plt.step(time, u * max(y), 'k:', label="Input")
plt.xlabel("Time (s)")
plt.ylabel("System Output")
plt.title("Generic First-Order System Simulation")
plt.grid(True)

################# System ID (ARX) ##########################
num_a = 1
num_b =1

arx = fl.ARX(num_a, num_b)
a_est, b_est, residuals = arx.fit(y, u)
A_est = float(a_est)
B_est = float(b_est)

sys_est = fl.StateSpaceSimulator(-a_est, b_est, C, D, Q=0, R=0, x0=[0])

y_est= []
for k in range(N):
    yk_est = sys_est.step(u[k])
    y_est.append(float(yk_est))


################ Kalman Filter ###############################
var_0 = 0
x_0 = 0
pvar = 0.08
mvar = 3

# Low pass and Kalman Filters
exp = fl.ExponentialFilter(0.1)
kf = fl.Kalman1D(var_0, x_0, pvar, mvar, -A_est, B_est)

y_lpf = []
x_kf = []
for k in range(N):
    kf.predict(u[k])
    kf.update(y[k])
    x_kf.append(float(kf.x))
    y_lpf.append(exp.update(y[k]))
    

plt.plot(time, x_kf, label="Kalman Filter Estimate")
plt.plot(time, y_lpf, label="LPF Estimate")
plt.plot(time, y_est, label="Estimated (ARX) Output")
plt.legend()
plt.show()