import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
from planning import randomized_search
from simulation import ik_solver


def init(X_GM_x: float = 0.0) -> state.Particle:
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.09], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.01], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0,
        X_GM,
        X_WO,
        "assets/big_fixed_puzzle.sdf",
        "assets/moving_puzzle.sdf",
        mu=0.6,
    )
    return p0


top_touch = set((("big_fixed_puzzle::b3", "block::201"),))


def try_refine_p():
    p0 = init()
    U = randomized_search.refine_p(p0, top_touch, components.stiff)
    print(f"{len(U)=}")


if __name__ == "__main__":
    try_refine_p()
