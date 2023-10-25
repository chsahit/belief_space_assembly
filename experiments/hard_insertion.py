import sys

import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import state
import utils
import visualize
from planning import refine_motion, search
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
    traj = search.refine_schedule(b, contact_defs.ground_align, modes)
    dynamics.visualize_trajectory(b.particles[0], traj, name="r0.html")
    dynamics.visualize_trajectory(b.particles[1], traj, name="r1.html")
    input()


def bilateral_noise_easy():
    p_a = init(X_GM_x=0.00, X_GM_p=-3.0)
    p_b = init(X_GM_x=0.00, X_GM_p=3.0)
    b = state.Belief([p_a, p_b])
    modes = [
        contact_defs.b_full_chamfer_touch,
        contact_defs.corner_touch,
        contact_defs.ground_align,
        contact_defs.ground_align,
    ]
    traj = search.refine_schedule(b, contact_defs.ground_align, modes)
    dynamics.visualize_trajectory(b.particles[0], traj, name="r0.html")
    dynamics.visualize_trajectory(b.particles[1], traj, name="r1.html")
    input()


if __name__ == "__main__":
    bilateral_noise()
