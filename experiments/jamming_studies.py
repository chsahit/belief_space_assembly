import pickle

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
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.34], [180, 0, 0])
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
        K = np.array([10.0, 100.0, 10.0, 600.0, 100.0, 100.0])
        X_WC_d = utils.xyz_rpy_deg([0.51, 0.000, 0.00], [180, -0, 0])
    elif i == 2:
        K = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 600.0])
        X_WC_d = utils.xyz_rpy_deg([0.5, 0.0, -0.00], [180, -10, 0])
    else:
        raise NotImplementedError
    X_GC = utils.xyz_rpy_deg([0.0, 0.0, 0.23], [0, 0, 0])
    return components.CompliantMotion(X_GC, X_WC_d, K, timeout=10.0)


def jamming():
    p1 = init(X_GM_x=0.0, X_GM_p=-10.0)
    u0 = init_motion(2)
    p11 = dynamics.simulate(p1, u0, vis=True)
    print(p11.contacts)


def planned_jamming():
    b0 = state.Belief([init(), init(X_GM_p=-10.0)])
    u = refine_motion.refine(b0, contact_defs.ground_align)


def drift():
    p0 = init()
    p0.env_geom = "assets/empty_world.sdf"
    X_WC_d = utils.xyz_rpy_deg([0.5, 0.0, 0.0], [180, 0, 0])
    X_GC = utils.xyz_rpy_deg([0.0, 0.0, 0.23], [0, 0, 0])
    u = components.CompliantMotion(X_GC, X_WC_d, components.stiff)

    dynamics.simulate(p0, u, vis=True)


def jacobian_analysis():
    def unit(v: np.ndarray) -> np.ndarray:
        return v / np.linalg.norm(v)

    with open("control_logs.pkl", "rb") as f:
        J_g, tau = pickle.load(f)
    # print(J_g.shape)
    F_d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1e-5])
    tau = J_g.T @ F_d
    v_q = 1e-20 * tau
    print(unit(J_g @ v_q))
    # tau_hat = np.linalg.pinv(J_g) @ np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1e-5])
    # print(np.linalg.pinv(J_g.T) @ tau_hat)


if __name__ == "__main__":
    planned_jamming()
