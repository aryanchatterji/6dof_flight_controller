import unittest
import numpy as np
from telemetry_dash import TelemetryParser, ConvergenceMonitor

# 1. MOCK TELEMETRY ENVIRONMENT

class TestTelemetryDashboard(unittest.TestCase):
    def setUp(self):
        self.mock_history = [
            {
                'time': 0.0,
                'position': np.array([0.0, 0.0, 100.0]),
                'velocity': np.array([10.0, 0.0, 0.0]),
                'quaternion': np.array([1.0, 0.0, 0.0, 0.0]),
                'convergence_status': False
            },
            {
                'time': 0.1,
                'position': np.array([1.0, 5.0, 99.9]),
                'velocity': np.array([10.0, 0.0, -1.0]),
                'quaternion': np.array([0.995, 0.0, 0.099, 0.0]),
                'convergence_status': True
            }
        ]
        self.parser = TelemetryParser(self.mock_history)

# 2. POSITIONAL ARRAY EXTRACTION

    def test_positional_extraction(self):
        x, y, z = self.parser.extract_positional_arrays()
        
        self.assertTrue(np.array_equal(x, [0.0, 1.0]), "X-axis extraction failed.")
        self.assertTrue(np.array_equal(y, [0.0, 5.0]), "Y-axis extraction failed.")
        self.assertTrue(np.array_equal(z, [100.0, 99.9]), "Z-axis extraction failed.")

# 3. SYSTEM CONVERGENCE RATIO

    def test_convergence_ratio_calculation(self):
        monitor = ConvergenceMonitor(self.parser)
        
        converged_steps = np.sum(monitor.convergence_states)
        total_steps = len(monitor.convergence_states)
        ratio = (converged_steps / total_steps) * 100
        
        self.assertEqual(ratio, 50.0, f"Convergence ratio calculated incorrectly. Expected 50.0, got {ratio}")

if __name__ == '__main__':
    unittest.main()