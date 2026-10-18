import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. TELEMETRY INGESTION ENGINE

class TelemetryParser:
    def __init__(self, data_history):
        self.raw_data = data_history
        self.time_array = np.array([state['time'] for state in self.raw_data])
        
    def extract_positional_arrays(self):
        positions = np.array([state['position'] for state in self.raw_data])
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]
        return x, y, z

    def extract_kinematic_arrays(self):
        velocities = np.array([state['velocity'] for state in self.raw_data])
        return velocities

# 2. 3D TRAJECTORY VISUALIZATION

class TrajectoryPlotter:
    def __init__(self, parser: TelemetryParser):
        self.parser = parser
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def render_flight_path(self):
        x, y, z = self.parser.extract_positional_arrays()
        
        self.ax.plot(x, y, z, label='6-DOF Flight Path', color='blue', linewidth=2)
        self.ax.scatter(x[0], y[0], z[0], color='green', label='Origin', s=50)
        self.ax.scatter(x[-1], y[-1], z[-1], color='red', label='Terminus', s=50)
        
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_zlabel('Altitude (m)')
        self.ax.set_title('Spatial Trajectory Telemetry')
        self.ax.legend()
        
        plt.tight_layout()
        plt.show()

# 3. ATTITUDE DYNAMICS PLOTTER

class AttitudeVisualizer:
    def __init__(self, parser: TelemetryParser):
        self.parser = parser
        
    def plot_euler_convergence(self):

        # Will handle conversion from quaternion arrays back to Euler for human-readable telemetry output.
        pass

# 4. SYSTEM CONVERGENCE DASHBOARD

class ConvergenceMonitor:
    def __init__(self, parser: TelemetryParser):
        self.parser = parser
        self.convergence_states = np.array([state.get('convergence_status', False) for state in self.parser.raw_data])
        
    def generate_convergence_report(self):
        converged_steps = np.sum(self.convergence_states)
        total_steps = len(self.convergence_states)
        ratio = (converged_steps / total_steps) * 100
        
        print("--- AEROELASTIC CONVERGENCE TELEMETRY ---")
        print(f"Total Simulation Steps: {total_steps}")
        print(f"Converged Steps: {converged_steps}")
        print(f"System Stability Ratio: {ratio:.2f}%")