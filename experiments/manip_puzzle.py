import time

import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
import visualize
from simulation import ik_solver


def init(X_WG_0_z: float = 0.3, X_GM_x: float = 0.0, mu: float = 0.0):
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, X_WG_0_z], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.09], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.01], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0,
        X_GM,
        X_WO,
        "assets/big_fixed_puzzle.sdf",
        "assets/moving_puzzle.sdf",
        mu=mu,
    )
    return p_0


def test_simulate():
    p_0 = init()
    t0 = time.time()
    X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.2], [180, 0, 0])
    X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.21], [180, 0, 7.8e-4])
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


def test_parallel_sim():
    p0 = init(X_GM_x=0.005)
    p1 = init(X_GM_x=-0.005)
    b = state.Belief([p0, p1])
    U = [
        components.CompliantMotion(
            RigidTransform(),
            utils.xyz_rpy_deg([0.5, 0.0, 0.20], [180, 0, 0]),
            components.stiff,
        ),
        components.CompliantMotion(
            RigidTransform(),
            utils.xyz_rpy_deg([0.51, 0.0, 0.20], [180, 0, 0]),
            components.stiff,
        ),
    ]
    visualize.play_motions_on_belief(b, U)
    input()


def n_motions():
    p = init(mu=0.6)
    X_WG_d0 = utils.xyz_rpy_deg([0.53, 0.0, 0.26], [180, 0, 0])
    u0 = components.CompliantMotion(RigidTransform(), X_WG_d0, components.stiff)
    p1 = dynamics.simulate(p, u0, vis=True)
    print(p1.sdf)
    print("sim 1")
    X_WC_d1 = utils.xyz_rpy_deg([0.49, 0.0, 0.08], [180, 0, 0])
    X_WC_d1 = utils.xyz_rpy_deg([0.46, 0.0, 0.23], [180, 0, 0])
    X_GC = RigidTransform([0.03, 0.0, 0.15])
    X_GC = RigidTransform()
    K1 = np.array([100.0, 10.0, 100.0, 600.0, 600.0, 600.0])
    u1 = components.CompliantMotion(X_GC, X_WC_d1, K1)
    p2 = dynamics.simulate(p1, u1, vis=True)
    print("sim 2")
    p3 = p2
    """
    X_WC_d2 = RigidTransform(p2.X_WG.rotation(), np.array([0.49, 0, 0.0]))
    K2 = np.array([100.0, 100.0, 100.0, 600.0, 600.0, 100.0])
    u2 = components.CompliantMotion(X_GC, X_WC_d2, K2)
    p3 = dynamics.simulate(p2, u2, vis=True)
    print("sim 3")
    """
    # X_WC_d2 = RigidTransform(p2.X_WG.rotation(), np.array([0.49, 0, 0.0]))
    X_WC_d3 = utils.xyz_rpy_deg([0.5, 0.0, 0.02], [180, 0, 0])
    X_WC_d3 = utils.xyz_rpy_deg([0.53, 0.0, 0.17], [180, 0, 0])
    K3 = np.array([100.0, 100.0, 100.0, 600.0, 600.0, 600.0])
    u3 = components.CompliantMotion(X_GC, X_WC_d3, K3)
    p4 = dynamics.simulate(p3, u3, vis=True)
    print("sim 4")

    input()


def poke():
    p_0 = init()
    # X_WG_d = utils.xyz_rpy_deg([0.50, 0.0, 0.2], [180, 0, 0])
    X_WG_d = utils.xyz_rpy_deg([0.501, 0.0, 0.28], [180, 0, 0])
    # X_WG_d = utils.xyz_rpy_deg([0.51288, 0.0, 0.27121], [180, 0, 0])
    u_0 = components.CompliantMotion(RigidTransform(), X_WG_d, components.stiff)
    p_1 = dynamics.simulate(p_0, u_0, vis=True)
    print(f"{p_1.epsilon_contacts()=}")
    input()


def test_ik():
    bottom = set((("fixed_puzzle::b1", "block::000"),))
    side = set(
        (
            ("fixed_puzzle::b1", "block::000"),
            ("fixed_puzzle::b2", "block::101"),
            ("fixed_puzzle::b2", "block::100"),
        )
    )
    ft = set((("fixed_puzzle::b3", "block::300"), ("fixed_puzzle::b3", "block::302")))
    ft = set((("fixed_puzzle::b3", "block::300"),))
    pose = ik_solver.project_manipuland_to_contacts(init(X_WG_0_z=0.29), ft)
    print("pose=")
    print(utils.rt_to_str(pose))


if __name__ == "__main__":
    test_parallel_sim()
