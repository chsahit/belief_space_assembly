import numpy as np
from pydrake.all import Quaternion, RigidTransform, RotationMatrix

import components
import dynamics
import state
import utils
import visualize
from simulation import ik_solver


def init(X_GM_x: float = 0.0):
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.155], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/big_chamfered_hole.sdf", "assets/peg.urdf"
    )
    return p_0


def do_sims():
    p0 = init()
    v = np.array([-0.01, 1, 0, 0])
    rot1_d = Quaternion(v / np.linalg.norm(v))
    X_WG_d = RigidTransform(rot1_d, [0.55, 0.0, 0.36])
    u0 = components.CompliantMotion(RigidTransform(), X_WG_d, components.stiff)
    dynamics.simulate(p0, u0, vis=True)
    print("done")
    input()


def joint_cartesian_tf():
    p0 = init()
    for i in range(6):
        init_K = np.copy(components.stiff)
        init_K[i] = components.soft[i]
        init_q_gains = np.diag((p0.J.T) @ np.diag(init_K) @ p0.J)
        X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
        u = components.CompliantMotion(RigidTransform(), X_WG_d, init_K)
        p1 = dynamics.simulate(p0, u, vis=False)
        final_K_gains = (
            np.linalg.pinv(p1.J.T) @ np.diag(init_q_gains) @ np.linalg.pinv(p1.J)
        )
        print(f"{init_K=}")
        print(f"{np.diag(final_K_gains)=}")
    input()


if __name__ == "__main__":
    joint_cartesian_tf()
