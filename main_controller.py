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

    # 3. CONTROL LOOP INTEGRATION


    def execute_control_step(self, target_quaternion):
        # 1. Retrieve raw telemetry from kinematics core
        current_quaternion = self.vehicle.orientation
        
        # 2. Calculate 3-axis rotational delta
        attitude_error = QuaternionControl.compute_error_vector(current_quaternion, target_quaternion)
        
        # 3. Feed error into PID loop to generate restorative torque commands
        corrective_moments = self.attitude_controller.compute_command(attitude_error, self.dt)
        
        # 4. Maintain baseline anti-gravity thrust (Z-axis assumed up)
        base_forces = np.array([0.0, 0.0, self.vehicle.mass * 9.81])
        
        # 5. Push commands to kinematics engine and execute time step
        state_telemetry = self.vehicle.run_simulation_step(self.dt, base_forces, corrective_moments)
        
        return state_telemetry


# 4. QUATERNION ATTITUDE ERROR COMPUTATION

class QuaternionControl:
    @staticmethod
    def compute_error_vector(q_current, q_target):
        # Compute conjugate of current state (q*)
        q_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        
        # Execute quaternion multiplication (q_target * q_conj) to find rotational delta
        w1, x1, y1, z1 = q_target
        w2, x2, y2, z2 = q_conj
        
        qw = w1*w2 - x1*x2 - y1*y2 - z1*z2
        qx = w1*x2 + x1*w2 + y1*z2 - z1*y2
        qy = w1*y2 - x1*z2 + y1*w2 + z1*x2
        qz = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        # Extract vector component representing axis-angle error
        error_vector = 2.0 * np.array([qx, qy, qz])
        
        # Enforce shortest path rotation mapping
        if qw < 0:
            error_vector = -error_vector
            
        return error_vector


# 5. FLIGHT SIMULATION EXECUTION LOOP

def run_flight_simulation(target_quat, duration, dt):
    # Initialize baseline rigid body state (mass in kg, inertia tensor)
    initial_state = QuaternionRigidBodyState(
        mass=1.5,
        inertia_tensor=np.diag([0.05, 0.05, 0.08])
    )
    
    # Initialize flight controller
    controller = FlightController(initial_state, dt)
    
    # Pre-allocate telemetry arrays for memory efficiency
    time_steps = int(duration / dt)
    history = {
        'time': np.zeros(time_steps),
        'orientation': np.zeros((time_steps, 4)),
        'angular_velocity': np.zeros((time_steps, 3))
    }
    
    # Execution loop
    for step in range(time_steps):
        current_time = step * dt
        
        # Execute active control step targeting the quaternion
        state_telemetry = controller.execute_control_step(target_quat)
        
        # Log active telemetry
        history['time'][step] = current_time
        history['orientation'][step] = state_telemetry['orientation']
        history['angular_velocity'][step] = state_telemetry['angular_velocity']
        
    return history
    

 