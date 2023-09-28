import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import state
import utils
from planning import refine_motion
from simulation import ik_solver


def init(X_GM_x: float = 0.0, X_GM_p: float = 0.0) -> state.Particle:
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.155], [0, X_GM_p, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/big_chamfered_hole.sdf", "assets/peg.urdf", mu=0.6
    )
    return p_0


def init_motion(i: int, K: np.ndarray = components.stiff) -> components.CompliantMotion:
    if i == 0:
        X_WC_d = utils.xyz_rpy_deg([0.52, 0.0, 0.10], [180, 0, 0])
    elif i == 1:
        # K = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 600.0])
        X_WC_d = utils.xyz_rpy_deg([0.505, 0.005, 0.01], [180, -5, 0])
    else:
        raise NotImplementedError
    X_GC = utils.xyz_rpy_deg([0.0, 0.0, 0.23], [0, 0, 0])
    return components.CompliantMotion(X_GC, X_WC_d, K)


def jamming():
    p0 = init()
    p1 = init(X_GM_x=0.0, X_GM_p=-10.0)
    u0 = init_motion(0)
    b0 = state.Belief([p0, p1])
    b1 = dynamics.f_bel(b0, u0)
    # p1 = dynamics.simulate(p1, u0, vis=True)
    assert b1.satisfies_contact(contact_defs.f_full_chamfer_touch)
    u1 = init_motion(1)
    b2 = dynamics.f_bel(b1, u1)
    print(b2.satisfies_contact(contact_defs.corner_align_2))
    px2 = dynamics.simulate(b1.particles[1], u1, vis=True)
    print(px2.contacts)


def drift():
    p0 = init()
    p0.env_geom = "assets/empty_world.sdf"
    X_WC_d = utils.xyz_rpy_deg([0.5, 0.0, 0.0], [180, 0, 0])
    X_GC = utils.xyz_rpy_deg([0.0, 0.0, 0.23], [0, 0, 0])
    u = components.CompliantMotion(X_GC, X_WC_d, components.stiff)

    dynamics.simulate(p0, u, vis=True)


if __name__ == "__main__":
    min_failure_mode()
