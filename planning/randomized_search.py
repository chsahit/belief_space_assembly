import logging
import random
from typing import List, Optional, Tuple

import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import sampler
import state
from planning import infer_joint_soln, stiffness

random.seed(1)
gen = np.random.default_rng(1)
np.set_printoptions(precision=3, suppress=True)
sample_logs = []
logger = logging.getLogger(__name__)
Data = Tuple[List[components.CompliantMotion], List[float]]


def compute_motions_from_particle(
    p: state.Particle,
    CF_d: components.ContactState,
    K: np.ndarray,
) -> Tuple[List[components.CompliantMotion], Data]:
    global sample_logs
    scores = []
    negative_motions = []
    targets = sampler.sample_from_contact(p, CF_d, num_samples=32, num_noise=32)
    sample_logs.append([target.multiply(p.X_GM) for target in targets])
    X_GC = RigidTransform([0, 0, 0.0])
    targets = [target.multiply(X_GC) for target in targets]
    # visualize.visualize_targets(p, targets)
    motions = [components.CompliantMotion(X_GC, target, K) for target in targets]
    # for m in motions:
    #     dynamics.simulate(p, m, vis=True)
    # breakpoint()
    if len(motions) == 0:
        return [], ([], [])
    P_next = dynamics.f_cspace(p, motions)
    U = []
    for i, p_next in enumerate(P_next):
        if p_next.satisfies_contact(CF_d):
            U.append(motions[i])
        else:
            negative_motions.append(motions[i])
            scores.append(0)
    if p.satisfies_contact(CF_d) and len(U) == 0:
        logger.info("warning, entire motion set loses contact")
    return U, (negative_motions, scores)


def compute_best_motion(
    b: state.Belief,
    CF_d: components.ContactState,
    K_star: np.ndarray,
    p_idx: int = 0,
) -> Tuple[components.CompliantMotion, float, bool, Data]:
    U0, data = compute_motions_from_particle(b.particles[p_idx], CF_d, K_star)
    if len(U0) == 0:
        return None, float("-inf"), False, ([], [])
    posteriors = dynamics.parallel_f_bel(b, U0)
    best_u = None
    most_certainty = float("-inf")
    pm_scores = []
    for p_i, posterior in enumerate(posteriors):
        certainty = posterior.partial_sat_score(CF_d)
        pm_scores.append(certainty)
        if certainty > most_certainty:
            most_certainty = certainty
            best_u = U0[p_i]
    success = most_certainty >= len(b.particles)
    data = (data[0] + U0, data[1] + pm_scores)
    return best_u, most_certainty, success, data


def iterative_gp(data, b, CF_d, do_gp, iters=3):
    max_certainty = float("-inf")
    best_u = None
    for idx in range(iters):
        # print(f"iteration={idx}")
        new_samples = infer_joint_soln.infer(*data, do_gp)
        posteriors = dynamics.parallel_f_bel(b, new_samples)
        scores = []
        for np_i, new_posterior in enumerate(posteriors):
            certainty = new_posterior.partial_sat_score(CF_d)
            p0_sat = new_posterior.particles[0].satisfies_contact(CF_d)
            p1_sat = new_posterior.particles[1].satisfies_contact(CF_d)
            is_partially_satisfiying = p0_sat or p1_sat
            if certainty > max_certainty and is_partially_satisfiying:
                max_certainty = certainty
                best_u = new_samples[np_i]
            if new_posterior.satisfies_contact(CF_d):
                return new_samples[np_i], certainty, True
            if is_partially_satisfiying:
                scores.append(1)
            else:
                scores.append(0)
        # print(f"{scores=}")
        data = (data[0] + new_samples, data[1] + scores)
    return best_u, max_certainty, False


def refine_b(
    b: state.Belief,
    CF_d: components.ContactState,
    search_compliance: bool,
    do_gp: bool = True,
) -> Optional[components.CompliantMotion]:
    if search_compliance:
        K_star = stiffness.solve_for_compliance(random.choice(b.particles))
    else:
        K_star = stiffness.ablate_compliance()
    logger.info(f"{np.diag(K_star)=}, ")
    best_u_root = None
    best_certainty_all = float("-inf")
    data = [[], []]
    for p_idx, p in enumerate(b.particles):
        best_u_i, certainty_i, success, data_i = compute_best_motion(
            b, CF_d, K_star, p_idx=p_idx
        )
        data[0] += data_i[0]
        data[1] += data_i[1]
        # print(f"certainty_{p_idx}={certainty_i}")
        if success:
            logger.info(f"certainty_{p_idx}={certainty_i}")
            return best_u_i
        if certainty_i > best_certainty_all:
            best_certainty_all = certainty_i
            best_u_root = best_u_i
    if best_u_root is None:
        logger.info("candidate motion sets are empty")
        return None
    u_gp, certainty_gp, success_gp = iterative_gp(data, b, CF_d, do_gp)
    if certainty_gp > best_certainty_all:
        logger.info(f"{certainty_gp=}")
        return u_gp
    logger.info(f"{best_certainty_all=}")
    return best_u_root
