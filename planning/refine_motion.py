import time
from typing import Dict, List, Tuple

import numpy as np
from pydrake.all import (
    Expression,
    MathematicalProgram,
    RigidTransform,
    RigidTransform_,
    Solve,
)
from tqdm import tqdm

import components
import dynamics
import state
import utils
from planning import directed_msets, motion_sets, randomized_search


def compute_compliance_frame(
    X_GM: RigidTransform,
    CF_d: components.ContactState,
    X_MP_dict: Dict[str, RigidTransform],
) -> RigidTransform:
    """Computes compliance frame X_GC for a desired contact CF_d.

    Computes a pose p_MC that minimizes expected torques about desired contacts. This
    is then re-rexpressed relative to the gripper by defining X_MC = (I_3, p_MC) and
    then X_GC = X_GM * X_MC. X_GM is some nominal grasp transform.

    Args:
        X_GM: RigidTransform describing a hypothesized manipuland grasp.
        CF_d: desired ContactState for the motion being refined.
        X_MP_dict: mapping from points on the manipuland which appear in CF_d to
        their pose on the the block.

    Returns:
        Compliance frame X_GC which tries to minimize the rotational displacement induced
        by linear errors about the contacts in CF_d.
    """

    def e3(i):
        v = np.zeros((3,))
        v[i] = 1
        return v

    prog = MathematicalProgram()
    p_MC = prog.NewContinuousVariables(3, "p_MC")
    X_MC = RigidTransform_[Expression](p_MC)
    for _, corner in CF_d:
        X_MP = RigidTransform_[Expression](X_MP_dict[corner].GetAsMatrix4())
        r = X_MC.translation() - X_MP.translation()
        for i in range(3):
            torque = np.cross(r, e3(i))
            prog.AddCost(torque.dot(torque))

    result = Solve(prog)
    assert result.is_success()
    X_MC_star = RigidTransform(result.GetSolution(p_MC))
    X_GM_rotated = RigidTransform(X_GM.translation())
    X_GC = X_GM_rotated.multiply(X_MC_star)
    print("(wrongish) compliance frame: ")
    print(utils.rt_to_str(X_GC))
    return X_GC


def compliance_search(
    X_GC: RigidTransform, CF_d: components.ContactState, p: state.Particle
) -> np.ndarray:
    print("searching for compliance")
    K_opt = components.stiff
    # U_opt = motion_sets.grow_motion_set(X_GC, K_opt, CF_d, p, density=5)
    r_vels = [
        directed_msets.gen.uniform(low=-0.05, high=0.05, size=3) for i in range(30)
    ]
    t_vels = [
        directed_msets.gen.uniform(low=-0.005, high=0.005, size=3) for i in range(30)
    ]
    U_opt = directed_msets.grow_randomized_mset(
        X_GC, K_opt, CF_d, p, density=5, r_vels=r_vels, t_vels=t_vels
    )
    print(f"K_curr={K_opt}, len(U_curr)={len(U_opt)}")
    for i in list(range(6)) + [2]:
        K_curr = K_opt.copy()
        K_curr[i] = components.soft[i]
        # U_curr = motion_sets.grow_motion_set(X_GC, K_curr, CF_d, p, density=5)
        U_curr = directed_msets.grow_randomized_mset(
            X_GC, K_curr, CF_d, p, density=5, r_vels=r_vels, t_vels=t_vels
        )
        print(f"{K_curr=}, {len(U_curr)=}")
        if len(U_curr) > len(U_opt):
            K_opt = K_curr
            U_opt = U_curr
    if len(U_opt) == 0:
        print("no compliance found")
        return None
    return K_opt


def refine(
    b0: state.Belief, CF_d: components.ContactState
) -> Tuple[components.CompliantMotion, state.Belief]:
    p_nom = b0.sample()
    p_nom = b0.particles[1]
    spheres = annotate_geoms.annotate(b0.particles[0].manip_geom)
    X_GC = compute_compliance_frame(p_nom.X_GM, CF_d, spheres)
    K_star = compliance_search(X_GC, CF_d, p_nom)
    if K_star is None:
        K_star = components.stiff
        # return None, None
    print(f"{K_star=}")
    # U_candidates = motion_sets.intersect_motion_sets(X_GC, K_star, b0, CF_d)
    U_candidates = [directed_msets.n_rrt(X_GC, K_star, b0, CF_d)]
    if U_candidates[0] == None:
        raise NotImplementedError("um, the search failed and idk what do")
    print("testing candidates")
    best_score = -1.0
    best_candidate = 0
    best_posterior = None
    for u_idx, u in enumerate(tqdm(U_candidates)):
        posterior = dynamics.f_bel(b0, u)
        if posterior.satisfies_contact(CF_d):
            print(f"sp = {utils.rt_to_str(u.X_WCd)}")
            return u, posterior
        else:
            score = posterior.score(CF_d)
            if score > best_score:
                best_score = score
                best_candidate = u_idx
                best_posterior = posterior
            print(f"{posterior.contact_state()=}")
    print(f"returning partial soln with score {best_score}")
    return U_candidates[best_candidate], best_posterior


def randomized_refine(
    b: state.Belief,
    modes: List[components.ContactState],
    search_compliance: bool = True,
    do_gp: bool = True,
    max_attempts: int = 3,
) -> List[components.CompliantMotion]:
    start_time = time.time()
    for attempt in range(max_attempts):
        print(f"{attempt=}")
        curr = b
        traj = []
        for mode in modes:
            print(f"{mode=}")
            m_best_score = 0.0
            while m_best_score < len(b.particles):
                u_star = randomized_search.refine_b(
                    curr, mode, search_compliance, do_gp
                )
                if u_star is None:
                    break
                curr_tenative = dynamics.f_bel(curr, u_star)
                curr_best_score = curr_tenative.partial_sat_score(mode)
                if curr_best_score <= m_best_score + 1e-4:
                    break
                m_best_score = curr_best_score
                curr = curr_tenative
                traj.append(u_star)
            if u_star is None:
                break
        if curr.satisfies_contact(modes[-1]):
            total_elapsed_time = time.time() - start_time
            sim_time = dynamics.get_time()
            np = dynamics.get_posterior_count()
            dynamics.reset_posteriors()
            dynamics.reset_time()
            return components.PlanningResult(traj, total_elapsed_time, sim_time, np)
    tet = time.time() - start_time
    sim_time, np = (dynamics.get_time(), dynamics.get_posterior_count())
    dynamics.reset_time()
    dynamics.reset_posteriors()
    return components.PlanningResult(None, tet, sim_time, np)
