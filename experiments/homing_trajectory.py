import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
from simulation import ik_solver


def make_homing_command(peg_urdf: str = "assets/peg.urdf"):
    # X_WG_0 = utils.xyz_rpy_deg([0.5025, 0.0025, 0.34], [180, 0, 0])
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([0, 0.0, 0.16], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.0], [0, 0, 0])
    # X_WO = utils.xyz_rpy_deg([0.5, y, 0.075], [0, 0, yaw])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0,
        X_GM,
        X_WO,
        "assets/real_hole.sdf",
        peg_urdf,
        mu=0.45,
    )
    K_large = np.array([100.0, 100.0, 40.0, 600.0, 600.0, 600.0])
    u0 = components.CompliantMotion(RigidTransform(), p0.X_WG, np.diag(K_large))
    utils.pickle_trajectory(p0, [u0], fname="logs/homing.pkl")
    dynamics.simulate(p0, u0, vis=True)
    input()


if __name__ == "__main__":
    make_homing_command()
