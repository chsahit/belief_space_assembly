import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
from planning import search
from simulation import ik_solver


def init() -> state.Belief:
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([0.0, 0.0, 0.09], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.01], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/fixed_puzzle.sdf", "assets/moving_puzzle.sdf"
    )
    return state.Belief([p0, p0])


bottom = set((("fixed_puzzle::b1", "block::000"),))


def puzzle_search():
    b0 = init()
    modes = [bottom, bottom]
    traj = search.refine_schedule(b0, bottom, modes)
    dynamics.visualize_trajectory(b0.particles[0], traj, name="p0.html")
    input()


if __name__ == "__main__":
    puzzle_search()
