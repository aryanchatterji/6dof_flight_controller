import unittest
import numpy as np
from main_controller import PIDController, ActuatorMixer, TrajectoryKinematics

# 1. PID SATURATION ENFORCEMENT

class TestMainController(unittest.TestCase):
    def test_pid_saturation_limits(self):
        pid = PIDController(
            kp=[10.0, 10.0, 10.0], 
            ki=[1.0, 1.0, 1.0], 
            kd=[1.0, 1.0, 1.0], 
            output_limits=(-50.0, 50.0)
        )
        
        extreme_error = np.array([1000.0, 1000.0, 1000.0])
        command = pid.compute_command(extreme_error, dt=0.01)
        
        self.assertTrue(
            np.all(command == 50.0), 
            f"PID controller failed to enforce upper saturation limits. Output: {command}"
        )
        
# 2. ACTUATOR MATRIX ALLOCATION

    def test_actuator_mixer_allocation(self):
        mixer = ActuatorMixer(motor_limits=(0.0, 100.0))
        
        total_thrust = 40.0
        moments = np.array([0.0, 0.0, 0.0])
        
        motor_cmds = mixer.compute_motor_outputs(total_thrust, moments)
        
        self.assertTrue(
            np.all(motor_cmds == 10.0), 
            f"Actuator mixer failed to distribute pure thrust symmetrically. Motor outputs: {motor_cmds}"
        )

# 3. THRUST TO QUATERNION MAPPING

    def test_thrust_to_quaternion_vertical(self):
        target_accel = np.array([0.0, 0.0, 9.81])
        target_yaw = 0.0
        
        quat, thrust = TrajectoryKinematics.acceleration_to_quaternion(target_accel, target_yaw)
        
        expected_quat = np.array([1.0, 0.0, 0.0, 0.0])
        error = np.linalg.norm(quat - expected_quat)
        
        self.assertTrue(
            error < 1e-6, 
            f"Kinematics failed to map vertical thrust to identity quaternion. Result: {quat}"
        )

if __name__ == '__main__':
    unittest.main()