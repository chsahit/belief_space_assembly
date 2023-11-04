import numpy as np
from pydrake.all import RigidTransform

import contact_defs
import state
import utils
from planning import directed_msets
from simulation import ik_solver


def init(X_GM_x: float = 0.0, X_GM_p: float = 0.0) -> RigidTransform:
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.155], [0, X_GM_p, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/big_chamfered_hole.sdf", "assets/peg.urdf", mu=0.6
    )
    return p_0


def n_rrt_check():
    p_a = init(X_GM_x=-0.01)
    p_b = init(X_GM_x=0.01)
    b = state.Belief([p_a, p_b])
    K_star = np.array([100.0, 100.0, 10.0, 100.0, 100.0, 600.0])
    X_GC = RigidTransform()
    u_star = directed_msets.n_rrt(b, K_star, X_GC, contact_defs.ground_align)


if __name__ == "__main__":
    n_rrt_check()
