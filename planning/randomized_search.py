import os
import random
import time
from typing import List

import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import mr
import puzzle_contact_defs
import state
from planning import infer_joint_soln
from simulation import generate_contact_set, ik_solver

random.seed(0)
gen = np.random.default_rng(0)


def reset_time():
    global scoring_time
    scoring_time = 0.0


if os.uname()[1] == "londonsystem" or True:
    compliance_samples = 16
    refinement_samples = 32
else:
    compliance_samples = 16
    refinement_samples = 32
print(f"{compliance_samples=}, {refinement_samples=}")


def apply_noise(targets: List[RigidTransform]) -> List[RigidTransform]:
    noised_targets = []
    for i, X in enumerate(targets):
        if i % 2 == 0:
            r_bounds = 0.05
            t_bounds = 0.01
        else:
            r_bounds = 0.001
            t_bounds = 0.0001
        r_vel = gen.uniform(low=-r_bounds, high=r_bounds, size=3)
        t_vel = gen.uniform(low=-t_bounds, high=t_bounds, size=3)
        random_vel = np.concatenate((r_vel, t_vel))
        X_noise = RigidTransform(mr.MatrixExp6(mr.VecTose3(random_vel)))
        noised_targets.append(X.multiply(X_noise))
    return noised_targets


def generate_targets(
    nominal: RigidTransform,
    count: int = 50,
    r_bound: float = 0.05,
    t_bound: float = 0.005,
) -> List[RigidTransform]:
    targets = [nominal]
    for i in range(count):
        r_vel = gen.uniform(low=-r_bound, high=r_bound, size=3)
        t_vel = gen.uniform(low=-t_bound, high=t_bound, size=3)
        random_vel = np.concatenate((r_vel, t_vel))
        tf = mr.MatrixExp6(mr.VecTose3(random_vel))
        sample = RigidTransform(nominal.GetAsMatrix4() @ tf)
        targets.append(sample)
    return targets


def solve_for_compliance(
    p: state.Particle, CF_d: components.ContactState
) -> np.ndarray:
    targets = generate_contact_set.project_manipuland_to_contacts(
        p, CF_d, num_samples=compliance_samples
    )
    targets = apply_noise(targets)
    K_opt = np.copy(components.stiff)
    validated_samples, _ = refine_p(p, CF_d, K_opt, targets=targets)
    succ_count = len(validated_samples)
    # print(f"{K_opt=}, {succ_count=}")
    if succ_count == len(targets):
        return K_opt, validated_samples
    for i in range(6):
        K_curr = np.copy(K_opt)
        K_curr[i] = components.soft[i]
        curr_validated_samples, _ = refine_p(p, CF_d, K_curr, targets=targets)
        curr_succ_count = len(curr_validated_samples)
        if curr_succ_count == len(targets):
            return K_curr, curr_validated_samples
        if curr_succ_count > succ_count:
            succ_count = curr_succ_count
            validated_samples = curr_validated_samples
            K_opt = K_curr
            # print(f"{K_opt=}, {succ_count=}")

    if succ_count == 0:
        K_opt_soft = np.copy(components.soft)
        validated_samples_soft, _ = refine_p(p, CF_d, K_opt_soft, targets=targets)
        succ_count_soft = len(validated_samples_soft)
        # print(f"{K_opt_soft=}, {succ_count_soft=}")
        for i in range(6):
            K_curr = np.copy(K_opt_soft)
            K_curr[i] = components.stiff[i]
            curr_validated_samples_soft, _ = refine_p(
                p, CF_d, K_opt_soft, targets=targets
            )
            curr_succ_count_soft = len(curr_validated_samples_soft)
            if curr_succ_count_soft > succ_count_soft:
                succ_count_soft = curr_succ_count_soft
                K_opt_soft = K_curr
                validated_samples_soft = curr_validated_samples_soft
                # print(f"{K_opt_soft=}, {succ_count_soft=}")
        if succ_count_soft > succ_count:
            K_opt = K_opt_soft
            validated_samples = validated_samples_soft
    return K_opt, validated_samples


def refine_p(
    p: state.Particle,
    CF_d: components.ContactState,
    K: np.ndarray,
    targets: List[RigidTransform] = None,
) -> List[components.CompliantMotion]:
    global scoring_time
    scores = []
    negative_motions = []
    nominal = p.X_WG
    if targets is None:
        targets = generate_contact_set.project_manipuland_to_contacts(
            p, CF_d, num_samples=refinement_samples
        )
        targets = apply_noise(targets)

    if "b2_right" in str(CF_d):
        X_GC = RigidTransform([0.0, -0.03, 0.0])
    else:
        X_GC = RigidTransform([0.0, 0.0, 0.15])
    targets = [target.multiply(X_GC) for target in targets]
    motions = [components.CompliantMotion(X_GC, target, K) for target in targets]
    motions = [ik_solver.update_motion_qd(m) for m in motions]
    # if np.linalg.norm(K - components.stiff) < 1e-3 and ("top" in str(CF_d)) and False:
    if np.linalg.norm(K - components.stiff) < 1e-3 and False:
        p_out = dynamics.simulate(p, motions[0], vis=True)
        print(f"{p_out.sdf=}")
    P_next = dynamics.f_cspace(p, motions)
    U = []
    s_time = time.time()
    for i, p_next in enumerate(P_next):
        if p_next.satisfies_contact(CF_d):
            U.append(motions[i])
        else:
            negative_motions.append(motions[i])
            scores.append(0)
    return U, (negative_motions, scores)


def score_tree_root(
    b: state.Belief,
    CF_d: components.CompliantMotion,
    K_star: np.ndarray,
    p_idx: int = 0,
    validated_samples=[],
) -> components.CompliantMotion:
    U0, data = refine_p(b.particles[p_idx], CF_d, K_star)
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
