import time

import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
import visualize
from simulation import ik_solver


def init(x: float = 0.0, pitch: float = 0.0):
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([x, 0.0, 0.155], [180, pitch, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/big_chamfered_hole.sdf", "assets/peg.urdf", mu=0.2
    )
    return p_0


def test_stiff():
    p_0 = init(pitch=3)
    X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.21], [180, 0, 0])
    stiff = np.array([100, 100, 100, 600, 600, 600])
    unilateral = np.array([10.0, 10.0, 30.0, 100, 100, 600.0])
    u_0 = components.CompliantMotion(RigidTransform(), X_WG_d, unilateral, timeout=10.0)
    p_1 = dynamics.simulate(p_0, u_0, vis=True)
    print(f"done")
    input()


def two_step_plan():
    p_0 = init(x=0.01)
    X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.23], [180, 0, 0])
    stiff = np.array([100, 100, 100, 600, 600, 600])
    unilateral = np.array([10.0, 10.0, 30.0, 100, 100, 600.0])
    u_0 = components.CompliantMotion(RigidTransform(), X_WG_d, unilateral, timeout=10.0)
    p_1 = dynamics.simulate(p_0, u_0, vis=True)
    X_WG_d2 = utils.xyz_rpy_deg([0.495, 0.0, 0.2], [180, 0, 0])
    u_1 = components.CompliantMotion(RigidTransform(), X_WG_d2, stiff, timeout=10.0)
    p_2 = dynamics.simulate(p_1, u_1, vis=True)
    print("done")
    input()


if __name__ == "__main__":
    two_step_plan()
