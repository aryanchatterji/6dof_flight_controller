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
         self.inv_inertia=np.lingalg.inv(self.inertia_matrix)

    def compute_kinetics(state,forces,moments):

         # 4. LINEAR KINETICS

         linear_acceleration=forces/state.mass

         # 5. ROTATIONAL KINETICS (EULER EQUATIONS)

         omega=state.angular_velocity
         gyroscopic_torque=np.cross(omega,state.inertia_matrix @ omega)
         angular_acceleration=state.inv_inertia @ (moments-gyroscopic_torque)

         # 6. KINETICS OUTPUT
    
         return state.velocity, linear_acceleration, angular_acceleration

