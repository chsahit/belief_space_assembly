import numpy as np
from pydrake.all import RigidTransform

import components
import utils
from belief import belief_state, dynamics
from simulation import ik_solver

X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0])
X_GB = utils.xyz_rpy_deg([0.0, 0.0, 0.155], [0, 0, 0])
X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
p_0 = belief_state.Particle(
    q_r_0, X_GB, X_WO, "assets/clean_bin.sdf", "assets/peg.urdf"
)
u_0 = components.CompliantMotion(RigidTransform(), X_WG_0, components.stiff)

dynamics.simulate(p_0, u_0, vis=True)
