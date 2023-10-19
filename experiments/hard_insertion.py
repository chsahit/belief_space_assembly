import sys

import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import state
import utils
import visualize
from planning import refine_motion
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


def init_motion(K: np.ndarray = components.stiff) -> components.CompliantMotion:
    X_WC_d = utils.xyz_rpy_deg([0.5, 0.0, 0.0], [180, 0, 0])
    X_GC = utils.xyz_rpy_deg([0.0, 0.0, 0.23], [0, 0, 0])
    return components.CompliantMotion(X_GC, X_WC_d, K)


def bilateral_noise():
    p_a = init(X_GM_x=0.00, X_GM_p=-10.0)
    p_b = init(X_GM_x=0.00, X_GM_p=10.0)
    b = state.Belief([p_a, p_b])
    modes = [
        contact_defs.b_full_chamfer_touch,
        contact_defs.corner_touch,
        contact_defs.corner_align_2,
        contact_defs.ground_align,
        contact_defs.ground_align,
    ]

    for mode in modes:
        u = refine_motion.refine(b, mode)
        if u is not None:
            b_next = dynamics.f_bel(b, u)
            for p in b.particles:
                dynamics.simulate(p, u, vis=True)
            b = b_next
        else:
            print("search failed")
            sys.exit()
    input()


if __name__ == "__main__":
    bilateral_noise()
