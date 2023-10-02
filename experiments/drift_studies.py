import pickle

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import RigidTransform
from tqdm import tqdm

import components
import contact_defs
import dynamics
import state
import utils
from simulation import ik_solver


def init(X_GM_x: float = 0.0, X_GM_p: float = 0.0) -> state.Particle:
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.34], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.155], [0, X_GM_p, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/empty_world.sdf", "assets/peg.urdf", mu=0.6
    )
    return p_0


def motion(B_ratio: float) -> components.CompliantMotion:
    X_GC = utils.xyz_rpy_deg([0.0, 0.0, 0.23], [0, 0, 0])
    X_WC_d = utils.xyz_rpy_deg([0.5, 0.0, -0.01], [180, 0, 0])
    B = B_ratio * np.sqrt(components.stiff)
    u = components.CompliantMotion(X_GC, X_WC_d, components.stiff, _B=B)
    return u


def test_damping_ratio():
    p0 = init()
    trajectories = []
    for b_ratio in tqdm(range(3, 12)):
        u = motion(b_ratio)
        pb = dynamics.simulate(p0, u)
        trajectories.append(pb.trajectory)
    with open("traj_logs.pkl", "wb") as f:
        pickle.dump(trajectories, f)


def plot_trajectories():
    with open("traj_logs.pkl", "rb") as f:
        trajectories = pickle.load(f)
    DRs = list(range(3, 12))
    cmap = [
        "r",
        "g",
        "b",
        "c",
        "m",
        "y",
        "k",
        "pink",
        "purple",
        "lime",
        "brown",
        "orange",
        "gray",
    ]
    for i, traj in enumerate(trajectories):
        print("plotting traj: ", i)
        for j in range(0, len(traj), 12):
            pt = traj[j]
            if j == 0:
                plt.scatter(pt[1][0], pt[1][2], c=cmap[i], label=str(DRs[i]))
            else:
                plt.scatter(pt[1][0], pt[1][2], c=cmap[i])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_damping_ratio()
    plot_trajectories()
