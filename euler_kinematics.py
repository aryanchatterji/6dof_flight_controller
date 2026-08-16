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
         self.inv_inertia=np.lingalg.inv

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
    def integrate_state(self,dt,linear_acceleration,angular_acceleration)

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

