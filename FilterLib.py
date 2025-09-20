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

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        