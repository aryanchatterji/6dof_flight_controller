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
        # Extract quaternions. Assumes format [qw, qx, qy, qz]
        
        quaternions = np.array([state['quaternion'] for state in self.parser.raw_data])
        
        qw = quaternions[:, 0]
        qx = quaternions[:, 1]
        qy = quaternions[:, 2]
        qz = quaternions[:, 3]
        
        # Quaternion to Euler Conversion (Roll, Pitch, Yaw)
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx**2 + qy**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
       
        pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (np.pi / 2), np.arcsin(sinp))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy**2 + qz**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        # Convert radians to degrees for readable dashboard output
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)
        
        # Attitude Dynamics
        plt.figure(figsize=(10, 6))
        plt.plot(self.parser.time_array, roll_deg, label='Roll ($\phi$)', color='blue', linewidth=1.5)
        plt.plot(self.parser.time_array, pitch_deg, label='Pitch ($\\theta$)', color='green', linewidth=1.5)
        plt.plot(self.parser.time_array, yaw_deg, label='Yaw ($\psi$)', color='red', linewidth=1.5)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Attitude Angle (Degrees)')
        plt.title('6DOF Attitude Telemetry: Euler Angle Convergence')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

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

# 5. MASTER EXECUTION PIPELINE

if __name__ == "__main__":
    # Test mock data structure 
    mock_history = [
        {
            'time': t,
            'position': [10*t, 5*np.sin(t), 1000 - 0.5*9.8*t**2],
            'velocity': [10, 5*np.cos(t), -9.8*t],
            'quaternion': [np.cos(t/2), np.sin(t/2)*0.1, np.sin(t/2)*0.05, 0], # Mock qw, qx, qy, qz
            'convergence_status': True if t > 2 else False
        } for t in np.linspace(0, 10, 100)
    ]
    
    # Initialize Engine
    telemetry = TelemetryParser(mock_history)
    
    # 1. Print Convergence Report
    monitor = ConvergenceMonitor(telemetry)
    monitor.generate_convergence_report()
    
    # 2. Render 3D Spatial Trajectory
    trajectory = TrajectoryPlotter(telemetry)
    trajectory.render_flight_path()
    
    # 3. Plot Attitude Dynamics (Quaternion to Euler)
    attitude = AttitudeVisualizer(telemetry)
    attitude.plot_euler_convergence()