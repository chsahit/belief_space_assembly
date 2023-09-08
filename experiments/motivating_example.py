import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
import visualize
from simulation import ik_solver


def init(z=0.155):
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([0.0, 0.0, z], [0, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/chamfered_hole.sdf", "assets/peg.urdf"
    )
    return p_0


def aligned_stiff_pih_no_rcc():
    p0 = init()
    X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.2], [180, 0, 0])
    u0 = components.CompliantMotion(RigidTransform(), X_WG_d, components.stiff)
    p1 = dynamics.simulate(p0, u0, vis=True)


if __name__ == "__main__":
    aligned_stiff_pih_no_rcc()
