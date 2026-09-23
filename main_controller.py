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


# 6. TRANSLATIONAL POSITION AND VELOCITY CONTROLLER

class TranslationalController:
    def __init__(self, pos_gains, vel_gains, vehicle_mass):
        self.pos_pid = PIDController(*pos_gains)
        self.vel_pid = PIDController(*vel_gains)
        self.mass = vehicle_mass
        self.gravity = np.array([0.0, 0.0, 9.81])

    def compute_target_acceleration(self, current_pos, target_pos, current_vel, target_vel, dt):
        # Outer loop: Position error generates target velocity
        pos_error = target_pos - current_pos
        commanded_vel = target_vel + self.pos_pid.compute_command(pos_error, dt)
        
        # Inner loop: Velocity error generates target acceleration
        vel_error = commanded_vel - current_vel
        target_accel = self.vel_pid.compute_command(vel_error, dt)
        
        # Feed-forward gravity compensation
        target_accel += self.gravity
        
        return target_accel

# 7. THRUST VECTOR TO QUATERNION MAPPING

class TrajectoryKinematics:
    @staticmethod
    def acceleration_to_quaternion(target_accel, target_yaw):
        # Normalize target acceleration to find desired body Z-axis
        thrust_norm = np.linalg.norm(target_accel)
        if thrust_norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0]), 0.0
            
        z_body = target_accel / thrust_norm
        
        # Define heading vector based on target yaw
        x_heading = np.array([np.cos(target_yaw), np.sin(target_yaw), 0.0])
        
        # Compute body Y-axis (cross product of Z-body and X-heading)
        y_body = np.cross(z_body, x_heading)
        y_norm = np.linalg.norm(y_body)
        
        if y_norm < 1e-6:
            # Handle singularity (pitch +/- 90 degrees)
            y_body = np.array([0.0, 1.0, 0.0])
        else:
            y_body /= y_norm
            
        # Compute true body X-axis to ensure orthonormal basis
        x_body = np.cross(y_body, z_body)
        
        # Construct rotation matrix
        R = np.column_stack((x_body, y_body, z_body))
        
        # Convert rotation matrix to quaternion (w, x, y, z)
        trace = np.trace(R)
        if trace > 0:
            S = 2.0 * np.sqrt(trace + 1.0)
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
            
        target_quat = np.array([qw, qx, qy, qz])
        target_quat /= np.linalg.norm(target_quat)
        
        return target_quat, thrust_norm

# 8. ACTUATOR MIXING MATRIX (QUAD-X CONFIGURATION)

class ActuatorMixer:
    def __init__(self, motor_limits=(0.0, 1.0)):
        self.limit_min, self.limit_max = motor_limits
        
        # Allocation Matrix for standard Quad-X config
        # Maps [Total Thrust, Roll Moment, Pitch Moment, Yaw Moment] to Motors [1, 2, 3, 4]
        self.allocation_matrix = np.array([
            [0.25, -0.25, -0.25, -0.25], # Motor 1 (Front Right, CCW)
            [0.25,  0.25,  0.25, -0.25], # Motor 2 (Rear Left, CCW)
            [0.25,  0.25, -0.25,  0.25], # Motor 3 (Front Left, CW)
            [0.25, -0.25,  0.25,  0.25]  # Motor 4 (Rear Right, CW)
        ])

    def compute_motor_outputs(self, total_thrust, moments):
        # Construct control vector: [F_z, M_x, M_y, M_z]
        control_vector = np.array([total_thrust, moments[0], moments[1], moments[2]])
        
        # Execute matrix multiplication to isolate motor thrust requirements
        motor_commands = np.dot(self.allocation_matrix, control_vector)
        
        # Enforce physical hardware saturation limits
        motor_commands = np.clip(motor_commands, self.limit_min, self.limit_max)
        
        return motor_commands


    

 