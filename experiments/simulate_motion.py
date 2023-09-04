import numpy as np
from pydrake.all import RigidTransform

import components
import utils
from belief import belief_state, dynamics
from simulation import ik_solver


def init():
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0])
    X_GB = utils.xyz_rpy_deg([0.0, 0.0, 0.155], [0, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = belief_state.Particle(
        q_r_0, X_GB, X_WO, "assets/clean_bin.sdf", "assets/peg.urdf"
    )
    return p_0


def test_simulate():
    p_0 = init()
    X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.2], [180, 0, 0])
    u_0 = components.CompliantMotion(RigidTransform(), X_WG_d, components.stiff)
    p_1 = dynamics.simulate(p_0, u_0, vis=True)
    print(f"{p_1.sdf=}")


def test_motion_set():
    p0 = init()


def test_belief_dynamics():
    pass


if __name__ == "__main__":
    test_motion_set()
