import numpy as np
from kinematics_core import QuaternionRigidBodyState, QuaternionMath

# 1. PROPORTIONAL-INTEGRAL-DERIVATIVE (PID) ARCHITECTURE

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(None, None)):
        self.kp = np.array(kp)
        self.ki = np.array(ki)
        self.kd = np.array(kd)
        
        self.integral_error = np.zeros(3)
        self.previous_error = np.zeros(3)
        
        self.limit_min, self.limit_max = output_limits

    def compute_command(self, error, dt):
        # Proportional calculation
        p_term = self.kp * error
        
        # Integral calculation 
        self.integral_error += error * dt
        i_term = self.ki * self.integral_error
        
        # Derivative calculation
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        self.previous_error = error
        
        output = p_term + i_term + d_term
        
        # Actuator saturation limits
        if self.limit_min is not None and self.limit_max is not None:
            output = np.clip(output, self.limit_min, self.limit_max)
            
        return output

# 2. MAIN FLIGHT CONTROLLER INITIALIZATION

class FlightController:
    def __init__(self, vehicle_state: QuaternionRigidBodyState, dt: float):
        self.vehicle = vehicle_state
        self.dt = dt
        
        # Initializing 3-axis rotational controllers (Roll, Pitch, Yaw)
        # Base gains initialized; strict tuning required in Phase 3
        self.attitude_controller = PIDController(
            kp=[15.0, 15.0, 15.0],
            ki=[0.1, 0.1, 0.1],
            kd=[5.0, 5.0, 5.0],
            output_limits=(-1000.0, 1000.0)
        )
        
        # Target telemetry arrays
        self.target_orientation_euler = np.zeros(3)
        self.target_body_rates = np.zeros(3)