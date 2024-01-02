import time
from typing import List

import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
import visualize
from planning import randomized_search, refine_motion
from puzzle_contact_defs import *
from simulation import diagram_factory, ik_solver


def init(X_GM_x: float = 0.0, X_GM_z: float = 0.0) -> state.Particle:
    z = 0.09 + X_GM_z
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.32], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, z], [180, 0, 0])
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
    if u_star is None:
        return b, None
    dynamics.simulate(b.particles[0], u_star, vis=True)
    dynamics.simulate(b.particles[1], u_star, vis=True)
    posterior = dynamics.f_bel(b, u_star)
    return posterior, u_star


def try_refine_b():
    p_a = init(X_GM_x=-0.0046666666)
    p_b = init(X_GM_x=0.004666666666)
    curr = state.Belief([p_a, p_b])
    diagram_factory.initialize_factory(curr.particles)
    modes = [top_touch2, bt, bt4, bottom, goal]
    traj, tet, st = refine_motion.refine_two_particles(curr, modes)
    """
    traj = []
    for mode in modes:
        curr, u_star = b_r(curr, mode)
        traj.append(u_star)
    """
    if traj is not None:
        visualize.play_motions_on_belief(state.Belief([p_a, p_b]), traj)
    input()


def nested_refine(
    b: state.Belief,
    CF_d: components.ContactState,
    modes: List[components.ContactState],
    max_attempts: int = 3,
) -> List[components.CompliantMotion]:
    start_time = time.time()
    for attempt in range(max_attempts):
        print(f"{attempt=}")
        curr = b
        traj = []
        for mode in modes:
            u_star = randomized_search.refine_b(curr, mode)
            if u_star is None:
                break
            traj.append(u_star)
            curr = dynamics.f_bel(curr, u_star)
        if curr.satisfies_contact(randomized_search.relax_CF(goal)):
            total_elapsed_time = time.time() - start_time
            sim_time = dynamics.get_time()
            dynamics.reset_time()
            return traj, total_elapsed_time, sim_time
    dynamics.reset_time()
    return None, -1.0, -1.0


def explore_x_preimg():
    stats = []
    deviations = np.linspace(0.002, 0.01, 4).tolist()
    max_attempts = 5
    # deviations = [0.0051]
    # max_attempts = 3
    for deviation in deviations:
        print(f"{deviation=}")
        fn = "full_plan_" + str(deviation)[2:6] + ".html"
        p_a = init(X_GM_x=-deviation)
        p_b = init(X_GM_x=deviation)
        b0 = state.Belief([p_a, p_b])
        modes = [top_touch2, bt, bt4, bottom, goal]
        traj, tet, st = nested_refine(b0, goal, modes, max_attempts=max_attempts)
        if traj is not None:
            visualize.play_motions_on_belief(b0, traj, fname=fn)
            stats.append(f"{deviation=} success")
        else:
            stats.append(f"{deviation=} failed")
    print("-----------")
    print(stats)


def explore_z_preimg():
    stats = []
    deviations = np.linspace(0.001, 0.0075, 4).tolist()
    deviations = [deviations[0]]
    max_attempts = 5
    for deviation in deviations:
        print(f"\n{deviation=}")
        fn = "z_full_plan_" + str(deviation)[2:6] + ".html"
        p_a = init(X_GM_z=-deviation)
        p_b = init(X_GM_z=deviation)
        b0 = state.Belief([p_a, p_b])
        diagram_factory.initialize_factory(b0.particles)
        modes = [top_touch2, bt, bt4, bottom, goal]
        traj, tet, st = nested_refine(b0, goal, modes, max_attempts=max_attempts)
        print(f"{tet=}, sim_time={st}")
        print(f"{randomized_search.scoring_time=}")
        randomized_search.reset_time()
        if traj is not None:
            visualize.play_motions_on_belief(b0, traj, fname=fn)
            stats.append(
                f"{deviation=} success, total_elapsed_time={tet}, sim_time={st}\n"
            )
        else:
            stats.append(
                f"{deviation=} failed, total_elapsed_time={tet}, sim_time={st}\n"
            )
    print("-----------")
    print(stats)


if __name__ == "__main__":
    try_refine_b()
