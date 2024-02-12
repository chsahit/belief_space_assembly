import os
import random
import time

import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import visualize
from planning import compliance_solver, infer_joint_soln
from simulation import generate_contact_set, ik_solver

random.seed(0)
gen = np.random.default_rng(0)


if os.uname()[1] == "londonsystem":
    compliance_samples = 16
    refinement_samples = 32
else:
    compliance_samples = 16
    refinement_samples = 32
print(f"{compliance_samples=}, {refinement_samples=}")


def solve_for_compliance(
    p: state.Particle, CF_d: components.ContactState
) -> np.ndarray:
    targets = generate_contact_set.project_manipuland_to_contacts(
        p, CF_d, num_samples=compliance_samples
    )
    """
    if "top" in str(CF_d):
        visualize.visualize_targets(p, targets)
    """
    # targets = apply_noise(targets)
    K_opt = np.copy(components.stiff)
    validated_samples, _ = compliance_solver.evaluate_K(p, CF_d, K_opt, targets=targets)
    succ_count = len(validated_samples)
    print(f"{K_opt=}, {succ_count=}")
    if succ_count == len(targets):
        return K_opt, validated_samples
    for i in range(6):
        K_curr = np.copy(K_opt)
        K_curr[i] = components.soft[i]
        curr_validated_samples, _ = compliance_solver.evaluate_K(
            p, CF_d, K_curr, targets=targets
        )
        curr_succ_count = len(curr_validated_samples)
        if curr_succ_count == len(targets):
            return K_curr, curr_validated_samples
        if curr_succ_count > succ_count:
            succ_count = curr_succ_count
            validated_samples = curr_validated_samples
            K_opt = K_curr
            print(f"{K_opt=}, {succ_count=}")

    if succ_count == 0:
        K_opt_soft = np.copy(components.soft)
        validated_samples_soft, _ = compliance_solver.evaluate_K(
            p, CF_d, K_opt_soft, targets=targets
        )
        succ_count_soft = len(validated_samples_soft)
        print(f"{K_opt_soft=}, {succ_count_soft=}")
        for i in range(6):
            K_curr = np.copy(K_opt_soft)
            K_curr[i] = components.stiff[i]
            curr_validated_samples_soft, _ = compliance_solver.evaluate_K(
                p, CF_d, K_opt_soft, targets=targets
            )
            curr_succ_count_soft = len(curr_validated_samples_soft)
            if curr_succ_count_soft > succ_count_soft:
                succ_count_soft = curr_succ_count_soft
                K_opt_soft = K_curr
                validated_samples_soft = curr_validated_samples_soft
                print(f"{K_opt_soft=}, {succ_count_soft=}")
        if succ_count_soft > succ_count:
            K_opt = K_opt_soft
            validated_samples = validated_samples_soft
    return K_opt, validated_samples


def score_tree_root(
    b: state.Belief,
    CF_d: components.CompliantMotion,
    K_star: np.ndarray,
    p_idx: int = 0,
    validated_samples=[],
) -> components.CompliantMotion:
    U0, data = compliance_solver.evaluate_K(b.particles[p_idx], CF_d, K_star)
    U0 = U0 + validated_samples
    if len(U0) == 0:
        return None, float("-inf"), False, ([], [])
    U0 = [ik_solver.update_motion_qd(u0) for u0 in U0]
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
        new_samples = [ik_solver.update_motion_qd(s) for s in new_samples]
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
) -> components.CompliantMotion:
    if search_compliance:
        K_star, samples = solve_for_compliance(random.choice(b.particles), CF_d)
        # print(f"{K_star=}, {len(samples)=}")
    else:
        K_star, samples = (components.stiff, [])
    best_u_root = None
    best_certainty_all = float("-inf")
    data = [[], []]
    for p_idx, p in enumerate(b.particles):
        if p_idx == 0:
            validated_samples = samples
        else:
            validated_samples = []
        best_u_i, certainty_i, success, data_i = score_tree_root(
            b, CF_d, K_star, p_idx=p_idx, validated_samples=validated_samples
        )
        data[0] += data_i[0]
        data[1] += data_i[1]
        # print(f"certainty_{p_idx}={certainty_i}")
        if success:
            print(f"{certainty_i=}")
            return best_u_i
        if certainty_i > best_certainty_all:
            best_certainty_all = certainty_i
            best_u_root = best_u_i
    if best_u_root is None:
        print("all candidate motion sets are empty")
        return None
    u_gp, certainty_gp, success_gp = iterative_gp(data, b, CF_d, do_gp)
    if certainty_gp > best_certainty_all:
        print(f"max certainty {certainty_gp} generated from gp")
        return u_gp
    print(f"{best_certainty_all=}")
    return best_u_root
