import numpy as np
from main_controller import FlightController
from kinematics_core import QuaternionRigidBodyState

# 1. TUNING ENVIRONMENT CONFIGURATION

class TuningEnvironment:
    def __init__(self, dt=0.01, sim_duration=5.0):
        self.dt = dt
        self.sim_duration = sim_duration
        self.time_steps = int(sim_duration / dt)
        
        # Standard step-response targets for tuning (45-degree roll command)
        self.target_quaternion = np.array([0.9238, 0.3827, 0.0, 0.0])
        
        # Baseline hardware parameters
        self.baseline_mass = 1.5
        self.baseline_inertia = np.diag([0.05, 0.05, 0.08])


# 2. COST FUNCTION ARCHITECTURE (ITAE)

class FitnessEvaluator:
    def __init__(self, env_config: TuningEnvironment):
        self.env = env_config

    def compute_itae_cost(self, kp, ki, kd):
        """
        Executes a closed-loop simulation using the provided gains and computes 
        the Integral Time-weighted Absolute Error (ITAE) to penalize late oscillations.
        """
        # Initialize pristine vehicle state for the trial iteration
        vehicle_state = QuaternionRigidBodyState(
            mass=self.env.baseline_mass,
            inertia_tensor=self.env.baseline_inertia
        )
        
        # Initialize isolated controller instance with the test gains
        controller = FlightController(vehicle_state, self.env.dt)
        controller.attitude_controller.kp = np.array(kp)
        controller.attitude_controller.ki = np.array(ki)
        controller.attitude_controller.kd = np.array(kd)
        
        total_cost = 0.0
        
        # Execute the step-response simulation loop
        for step in range(self.env.time_steps):
            time = step * self.env.dt
            
            # Run active control step targeting the step-response quaternion
            state_telemetry = controller.execute_control_step(self.env.target_quaternion)
            current_quat = state_telemetry['orientation']
            
            # Calculate angular error magnitude
            error_mag = np.linalg.norm(current_quat[1:4] - self.env.target_quaternion[1:4])
            
            # ITAE penalty mapping: Time * Absolute Error
            total_cost += time * error_mag * self.env.dt
            
        return total_cost