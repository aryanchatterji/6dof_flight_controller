import unittest
import numpy as np
from kinematics_core import QuaternionRigidBodyState, ClimbScenarios
from main_controller import FlightController
from mdo_pipeline import MDOPipelineEnvironment, AeroelasticSolver

# 1. QUATERNION SINGULARITY VALIDATION

class TestQuaternionKinematics(unittest.TestCase):
    def setUp(self):
        self.mass = 1500.0
        self.inertia = np.diag([25000.0, 30000.0, 45000.0])
        self.dt = 0.01
        self.steps = 1500
        
    def test_vertical_climb_singularity(self):
        # Initialize quaternion state to replace flawed Euler architecture
        vehicle = QuaternionRigidBodyState(self.mass, self.inertia)
        forces, moments = ClimbScenarios.generate_vertical_test(self.steps, self.dt)
        
        nan_encountered = False
        for i in range(self.steps):
            state = vehicle.run_simulation_step(self.dt, forces[i], moments[i])
            if np.any(np.isnan(state["quaternion"])) or np.any(np.isnan(state["euler_angles"])):
                nan_encountered = True
                break
                
        self.assertFalse(nan_encountered, "Division by zero (NaN) encountered. Gimbal lock persists.")

# 2. PSO GAIN CONVERGENCE RESPONSE

class TestFlightControllerStability(unittest.TestCase):
    def setUp(self):
        self.mass = 1.5
        self.inertia = np.diag([0.05, 0.05, 0.08])
        self.dt = 0.01
        self.vehicle = QuaternionRigidBodyState(self.mass, self.inertia)
        self.controller = FlightController(self.vehicle, self.dt)
        
        # Simulated PSO optimal gains
        self.controller.attitude_controller.kp = np.array([5.0, 5.0, 5.0])
        self.controller.attitude_controller.ki = np.array([0.1, 0.1, 0.1])
        self.controller.attitude_controller.kd = np.array([2.5, 2.5, 2.5])
        
    def test_roll_step_response(self):
        # 45-degree roll target
        target_quat = np.array([0.9238, 0.3827, 0.0, 0.0])
        steps = 500
        
        for _ in range(steps):
            telemetry = self.controller.execute_control_step(target_quat)
            
        # Assert system reaches steady-state near target
        final_quat = telemetry["quaternion"]
        error_mag = np.linalg.norm(final_quat[1:4] - target_quat[1:4])
        
        self.assertTrue(error_mag < 0.05, f"System failed to converge. Final error magnitude: {error_mag}")

# 3. AEROELASTIC EQUILIBRIUM LIMIT TEST

class TestAeroelasticSolver(unittest.TestCase):
    def setUp(self):
        self.env = MDOPipelineEnvironment()
        # Override for extreme stress testing
        self.env.max_coupling_iterations = 150 
        self.solver = AeroelasticSolver(self.env)
        
    def test_extreme_dynamic_pressure_convergence(self):
        # Inject extreme Mach 2+ velocity vector to maximize dynamic pressure
        extreme_velocity = np.array([800.0, 0.0, 0.0]) 
        
        # Execute coupling loop
        converged = self.solver.run_coupling_iteration(extreme_velocity)
        
        self.assertTrue(converged, "Solver failed to reach static equilibrium within 150 iterations under extreme load.")

if __name__ == '__main__':
    unittest.main()