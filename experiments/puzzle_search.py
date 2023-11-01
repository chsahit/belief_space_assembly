import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
from planning import search
from simulation import ik_solver


def init(X_GM_x: float = 0.0) -> state.Particle:
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.09], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.01], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/fixed_puzzle.sdf", "assets/moving_puzzle.sdf"
    )
    return p0


bottom = set((("fixed_puzzle::b1", "block::000"),))
side = set(
    (
        ("fixed_puzzle::b1", "block::000"),
        ("fixed_puzzle::b2", "block::101"),
        ("fixed_puzzle::b2", "block::100"),
    )
)


def puzzle_search():
    p_a = init(X_GM_x=-0.005)
    p_b = init(X_GM_x=0.005)
    b0 = state.Belief([p_a, p_b])
    modes = [bottom, side]
    traj = search.refine_schedule(b0, bottom, modes)
    dynamics.visualize_trajectory(b0.particles[0], traj, name="p0.html")
    input()


if __name__ == "__main__":
    puzzle_search()
