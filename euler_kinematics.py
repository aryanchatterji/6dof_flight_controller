import numpy as np

class RigidBodyState:
    def __init__(self,mass,inertia_tensor):

         # 1. INITIALIZED TRANSLATIONAL VECTORS

         self.position=np.zeros(3)
         self.velocity=np.zeros(3)

         # 2. INITIALIZED ROTATIONAL VECTORS WITH EULER ANGLES

         # [phi, theta, psi] representing roll, pitch, and yaw
         self.orientation=np.zeros(3)
         self.angular_velocity=np.zeros(3)

         # 3. MASS AND INERTIA MAPPING

         self.mass=mass
         self.inertia_matrix=np.array(inertia_tensor)

         # Pre-computed inverse to save loop processing power
         self.inv_inertia = np.linalg.inv(self.inertia_matrix)

    def compute_kinetics(state,forces,moments):

         # 4. LINEAR KINETICS

         linear_acceleration=forces/state.mass

         # 5. ROTATIONAL KINETICS (EULER EQUATIONS)

         omega=state.angular_velocity
         gyroscopic_torque=np.cross(omega,state.inertia_matrix @ omega)
         angular_acceleration=state.inv_inertia @ (moments-gyroscopic_torque)

         # 6. KINETICS OUTPUT
    
         return state.velocity, linear_acceleration, angular_acceleration


    # 7. NUMERICAL INTEGRATION LOOP (EULER METHOD)
    
    def integrate_state(self,dt,linear_acceleration,angular_acceleration):

         # Translational state update
             self.velocity += linear_acceleration * dt
             self.position += self.velocity * dt

             # Rotational state update (body-frame angular velocity)
             self.angular_velocity += angular_acceleration * dt

             # 8. EULER ANGLE KINEMATICS MATRIX (NON-LINEAR TRANSFORMATION)
             
             phi, theta, psi = self.orientation
             p, q, r = self.angular_velocity

             # Mapping body-frame rates (p,q,r) to inertial Euler rates (phi_dot, theta_dot, psi_dot)
             phi_dot = p + np.sin(phi) * np.tan(theta) * q + np.cos(phi) * np.tan(theta) * r
             theta_dot = np.cos(phi) * q - np.sin(phi) * r
             psi_dot = (np.sin(phi) / np.cos(theta)) * q + (np.cos(phi) / np.cos(theta)) * r
             
             # Update orientation
             self.orientation += np.array([phi_dot, theta_dot, psi_dot]) * dt

     # 9. DISCRETE SIMULATION STEP

    def run_simulation_step(self, dt, forces, moments):

         # Executing kinetic derivative calculations
         _, linear_accel, angular_accel = self.compute_kinetics(forces, moments)

         # Executing numerical integration over time step dt
         self.integrate_state(dt, linear_accel, angular_accel)

         # 10. STATE TELEMETRY EXPORT

         return {
             "position": np.copy(self.position),
             "velocity": np.copy(self.velocity),
             "euler_angles": np.copy(self.orientation),
             "body_rates": np.copy(self.angular_velocity)
         }
    
# 11. SIMULATION ENVIRONMENT INITIALIZATION

class FlightSimulation:
      def __init__(self, rigid_body: RigidBodyState, dt: float, duration: float):
            self.vehicle=rigid_body
            self.dt=dt
            self.steps=int(duration/dt)

            # 12. TELEMETRY DATA PRE-ALLOCATION

            self.time_series=np.zeros(self.steps)
            self.position_log=np.zeros((self.steps,3))
            self.orientation_log=np.zeros((self.steps,3))


     # 13. TIME-MARCHING EXECUTION LOOP

      def run_scenario(self, force_array, moment_array):
            for i in range(self.steps):
                  current_time=i*self.dt

                  # Execute discrete time step
                  state=self.vehicle.run_simulation_step(self.dt, force_array[i], moment_array[i])

                  # Log discrete state variables
                  self.time_series[i]=current_time
                  self.position_log[i]=state["position"]
                  self.orientation_log[i]=state["euler_angles"]

            return self.time_series, self.position_log, self.orientation_log

# 14. FLIGHT TEST SCENARIOS

class TestScenarios:
    @staticmethod
    def generate_pitch_maneuver(steps, dt):
        force_array = np.zeros((steps, 3))
        moment_array = np.zeros((steps, 3))

        # Applying a constant pitch moment to verify 
        # rotational integration loops.
        for i in range(steps):
            moment_array[i] = np.array([0.0, 50.0, 0.0])

        return force_array, moment_array

# 15. KINEMATIC SCENARIO EXECUTION

if __name__ == "__main__":
    # Standard rigid body init 
    mass = 1500.0
    inertia = [
        [25000.0, 0.0, 0.0],
        [0.0, 30000.0, 0.0],
        [0.0, 0.0, 45000.0]
    ]
    
    vehicle_state = RigidBodyState(mass, inertia)
    
    # Sim parameters
    dt = 0.01
    duration = 15.0 
    
    sim_env = FlightSimulation(vehicle_state, dt, duration)
    forces, moments = TestScenarios.generate_pitch_maneuver(sim_env.steps, dt)
    
    # 16. TIME-MARCHING EXECUTION
    
    # Running the scenario to log telemetry 
    time_data, pos_data, ori_data = sim_env.run_scenario(forces, moments)
    print("Test maneuver complete. Ready for telemetry plotting.")

# 17. SYSTEM TELEMETRY MONITOR

class TelemetryMonitor:
    def __init__(self, pitch_limit_deg=89.5):
        self.pitch_limit = np.radians(pitch_limit_deg)

    def check_state_validity(self, current_time, orientation):
        phi, theta, psi = orientation
        
        # Standard telemetry integrity check for floating-point errors
        if np.isnan(phi) or np.isnan(theta) or np.isnan(psi):
            print(f"[ERROR] T={current_time:.2f}s : NaN encountered in orientation array.")
            return False
            
        # Flight envelope monitor: alert if pitch exceeds standard vertical bounds
        if np.abs(theta) >= self.pitch_limit:
            print(f"[WARNING] T={current_time:.2f}s : Pitch angle exceeds test envelope.")
            return False
            
        return True

# 18. VERTICAL CLIMB TEST SCENARIO

class ClimbScenarios:
    @staticmethod
    def generate_vertical_test(steps, dt):
        force_array = np.zeros((steps, 3))
        moment_array = np.zeros((steps, 3))

        # Apply sustained pitch moment to evaluate vertical climb envelope
        for i in range(steps):
            force_array[i] = np.array([10000.0, 0.0, 0.0])
            moment_array[i] = np.array([0.0, 1500.0, 0.0])

        return force_array, moment_array

# 19. MONITORED EXECUTION LOOP

class MonitoredFlightSimulation(FlightSimulation):
    def __init__(self, rigid_body, dt, duration):
        super().__init__(rigid_body, dt, duration)
        self.monitor = TelemetryMonitor()

    def run_monitored_scenario(self, force_array, moment_array):
        for i in range(self.steps):
            current_time = i * self.dt
            
            state = self.vehicle.run_simulation_step(self.dt, force_array[i], moment_array[i])
            
            self.time_series[i] = current_time
            self.position_log[i] = state["position"]
            self.orientation_log[i] = state["euler_angles"]

            # Halt simulation if state arrays corrupt
            integrity_check = self.monitor.check_state_validity(current_time, state["euler_angles"])
            if not integrity_check:
                print(f"[SYSTEM] Simulation halted at Step {i}. Truncating telemetry arrays.")
                return (
                    self.time_series[:i], 
                    self.position_log[:i], 
                    self.orientation_log[:i]
                )

        return self.time_series, self.position_log, self.orientation_log

    def check_state_validity(self, current_time, orientation):
        phi, theta, psi = orientation
        
        # Standard telemetry integrity check for floating-point errors
        if np.isnan(phi) or np.isnan(theta) or np.isnan(psi):
            print(f"[CRITICAL ERROR] T={current_time:.2f}s : NaN encountered in orientation array.")
            # TODO (Bug #01): Root cause identified as Euler angle singularity at pitch = 90 deg.
            # Kinematic matrix divides by zero. 
            # Halting flight controller development to research non-singular rotational architectures.
            return False
            
        # Flight envelope monitor: alert if pitch exceeds standard vertical bounds
        if np.abs(theta) >= self.pitch_limit:
            print(f"[WARNING] T={current_time:.2f}s : Pitch angle exceeds test envelope.")
            return False
            
        return True

# 20. ARCHITECTURE ROADMAP AND KNOWN ISSUES

# [BUG 01] EULER SINGULARITY (GIMBAL LOCK)
# STATUS: Unresolved. Active Research Phase.
# DESCRIPTION: The 3-axis kinematic integration encounters a division-by-zero
# error during 90-degree vertical pitch maneuvers. 
# ACTION ITEM: Halt Euler kinematic expansion. Research and implement 
# 4-dimensional Quaternion arrays to bypass gimbal lock limitations.
# TARGET: Phase 2 Integration.

# [UPDATE] DEPRECATION NOTICE
# RigidBodyState Euler integration is formally marked for deprecation.
# Quaternion mathematical formulation is actively ongoing. Euler kinematics 
# will be retained temporarily solely as a benchmark baseline for the new architecture.