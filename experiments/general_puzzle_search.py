import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
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
        "assets/big_fixed_puzzle_div.sdf",
        "assets/moving_puzzle.sdf",
        mu=0.6,
    )
    return p0


top_touch = set((("big_fixed_puzzle::b3_out", "block::b4"),))
ft = set(
    (
        ("big_fixed_puzzle::b3", "block::b4"),
        ("big_fixed_puzzle::b3", "block::b4"),
    )
)
it = set(
    (
        ("big_fixed_puzzle::b3_in", "block::b3"),
        ("big_fixed_puzzle::b3_in", "block::b3"),
    )
)
bt = set((("big_fixed_puzzle::b4_in", "block::b5"),))
it2 = set((("big_fixed_puzzle::b4_in", "block::b3"),))


def try_refine_p():
    p0 = init()
    U = randomized_search.refine_p(p0, top_touch, components.stiff)
    print(f"{len(U)=}")


def b_r(b, mode):
    u_star = randomized_search.refine_b(b, mode)
    dynamics.simulate(b.particles[0], u_star, vis=True)
    dynamics.simulate(b.particles[1], u_star, vis=True)
    posterior = dynamics.f_bel(b, u_star)
    return posterior


def try_refine_b():
    p_a = init(X_GM_x=-0.005)
    p_b = init(X_GM_x=0.005)
    b0 = state.Belief([p_a, p_b])
    b1 = b_r(b0, top_touch)
    b2 = b_r(b1, it2)
    b3 = b_r(b1, bt)
    b4 = b_r(b3, it)

    """
    u_star = randomized_search.refine_b(b0, top_touch)
    dynamics.simulate(p_a, u_star, vis=True)
    dynamics.simulate(p_b, u_star, vis=True)
    b1 = dynamics.f_bel(b0, u_star)
    u1_star = randomized_search.refine_b(b1, it2)
    dynamics.simulate(b1.particles[0], u1_star, vis=True)
    dynamics.simulate(b1.particles[1], u1_star, vis=True)

    b1 = dynamics.f_bel(b0, u_star)
    u1_star = randomized_search.refine_b(b1, it2)
    dynamics.simulate(b1.particles[0], u1_star, vis=True)
    dynamics.simulate(b1.particles[1], u1_star, vis=True)
    b2 = dynamics.f_bel(b1, u1_star)
    u2_star = randomized_search.refine_b(b2, it)
    dynamics.simulate(b2.particles[0], u2_star, vis=True)
    dynamics.simulate(b2.particles[1], u2_star, vis=True)
    """
    input()


def try_refine_b_pih():
    p_a = init_pih(X_GM_x=-0.005)
    p_b = init_pih(X_GM_x=-0.0049)
    b0 = state.Belief([p_a, p_b])
    u_star = randomized_search.refine_b(b0, contact_defs.b_full_chamfer_touch)
    if u_star is not None:
        print(f"{u_star.X_WCd=}")
    else:
        return None
    b1 = dynamics.f_bel(b0, u_star)
    u1_star = randomized_search.refine_b(b1, contact_defs.ground_align)
    b2 = dynamics.f_bel(b1, u1_star)
    u2_star = randomized_search.refine_b(b2, contact_defs.ground_align)
    for p in b2.particles:
        dynamics.simulate(p, u2_star, vis=True)


if __name__ == "__main__":
    try_refine_b()
