import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import state
import utils
import visualize
from planning import randomized_search
from puzzle_contact_defs import *
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


def try_refine_p():
    p0 = init()
    U = randomized_search.refine_p(p0, top_touch, components.stiff)
    print(f"{len(U)=}")


def b_r(b, mode):
    u_star = randomized_search.refine_b(b, mode)
    dynamics.simulate(b.particles[0], u_star, vis=True)
    dynamics.simulate(b.particles[1], u_star, vis=True)
    posterior = dynamics.f_bel(b, u_star)
    return posterior, u_star


def try_refine_b():
    p_a = init(X_GM_x=-0.005)
    p_b = init(X_GM_x=0.005)
    curr = state.Belief([p_a, p_b])
    modes = [top_touch2, bt, bt4, bottom, goal]
    traj = []
    for mode in modes:
        curr, u_star = b_r(curr, mode)
        traj.append(u_star)
    visualize.play_motions_on_belief(state.Belief([p_a, p_b]), traj)
    input()


if __name__ == "__main__":
    try_refine_b()
