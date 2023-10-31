import time

import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
from simulation import ik_solver


def init():
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([0.0, 0.0, 0.09], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.01], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/fixed_puzzle.sdf", "assets/moving_puzzle.sdf"
    )
    return p_0


def test_simulate():
    p_0 = init()
    t0 = time.time()
    X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.2], [180, 0, 0])
    u_0 = components.CompliantMotion(RigidTransform(), X_WG_d, components.stiff)
    p_1 = dynamics.simulate(p_0, u_0, vis=True)
    print(f"{p_1.epsilon_contacts()=}")
    X_WG_d1 = utils.xyz_rpy_deg([0.5, 0.03, 0.2], [180, 0, 0])
    K_l = np.array([100.0, 100.0, 100.0, 100.0, 600.0, 100.0])
    u_1 = components.CompliantMotion(RigidTransform(), X_WG_d1, K_l)
    p_2 = dynamics.simulate(p_1, u_1, vis=True)
    tT = time.time()
    print(f"{p_2.epsilon_contacts()=}")
    print(f"sim time={tT - t0}")
    input()


def test_ik():
    bottom = set((("fixed_puzzle::b1", "block::000"),))
    pose = ik_solver.project_manipuland_to_contacts(init(), bottom)

if __name__ == "__main__":
    test_ik()
