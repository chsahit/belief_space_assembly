# IK solving is bad, this exists now
import os
from typing import List

import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import mr
import puzzle_contact_defs

print("Warning, hardcoded dependency on puzzle_contact_defs")
import state
from simulation import generate_contact_set, ik_solver

gen = np.random.default_rng(0)

if os.uname()[1] == "londonsystem":
    compliance_samples = 16
    refinement_samples = 32
else:
    compliance_samples = 40
    refinement_samples = 60
print(f"{compliance_samples=}, {refinement_samples=}")


def relax_CF(CF_d: components.ContactState) -> components.ContactState:
    relaxation = puzzle_contact_defs.relaxations.get(frozenset(CF_d), None)
    if relaxation is not None:
        return relaxation
    relaxed_CF_d = set()
    for env_contact, manip_contact in CF_d:
        e_u = env_contact.rfind("_")
        r_ec = env_contact[:e_u]
        e_m = manip_contact.rfind("_")
        r_mc = manip_contact[:e_m]
        relaxed_CF_d.add((r_ec, r_mc))
    return relaxed_CF_d


def apply_noise(targets: List[RigidTransform]) -> List[RigidTransform]:
    noised_targets = []
    for X in targets:
        r_vel = gen.uniform(low=-0.05, high=0.05, size=3)
        t_vel = gen.uniform(low=-0.01, high=0.01, size=3)
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
    validated_samples = refine_p(p, CF_d, K_opt, targets=targets)
    succ_count = len(validated_samples)
    print(f"{K_opt=}, {succ_count=}")
    if succ_count == len(targets):
        return K_opt, validated_samples
    for i in range(6):
        K_curr = np.copy(K_opt)
        K_curr[i] = components.soft[i]
        curr_validated_samples = refine_p(p, CF_d, K_curr, targets=targets)
        curr_succ_count = len(curr_validated_samples)
        if curr_succ_count == len(targets):
            print(f"K_opt={K_curr}")
            return K_curr, curr_validated_samples
        if curr_succ_count > succ_count:
            succ_count = curr_succ_count
            validated_samples = curr_validated_samples
            K_opt = K_curr
            print(f"{K_opt=}, {succ_count=}")

    if succ_count == 0:
        K_opt_soft = np.copy(components.soft)
        validated_samples_soft = refine_p(p, CF_d, K_opt_soft, targets=targets)
        succ_count_soft = len(validated_samples_soft)
        print(f"{K_opt_soft=}, {succ_count_soft=}")
        for i in range(6):
            K_curr = np.copy(K_opt_soft)
            K_curr[i] = components.stiff[i]
            curr_validated_samples_soft = refine_p(p, CF_d, K_opt_soft, targets=targets)
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


def refine_p(
    p: state.Particle,
    CF_d: components.ContactState,
    K: np.ndarray,
    targets: List[RigidTransform] = None,
) -> List[components.CompliantMotion]:
    nominal = p.X_WG
    if targets is None:
        targets = generate_contact_set.project_manipuland_to_contacts(
            p, CF_d, num_samples=refinement_samples
        )
        targets = apply_noise(targets)

    X_GC = RigidTransform([0.0, 0.0, 0.15])
    targets = [target.multiply(X_GC) for target in targets]
    motions = [components.CompliantMotion(X_GC, target, K) for target in targets]
    # if np.linalg.norm(K - components.stiff) < 1e-3 and ("b3" in str(CF_d)):
    if abs(K[1] - 10) < 1e-3 and ("b3" in str(CF_d)) and False:
        dynamics.simulate(p, motions[0], vis=True)
    P_next = dynamics.f_cspace(p, motions)
    U = []
    relaxed_CF_d = relax_CF(CF_d)
    for i, p_next in enumerate(P_next):
        if p_next.satisfies_contact(relaxed_CF_d):
            U.append(motions[i])
    return U


def score_tree_root(
    b: state.Belief,
    CF_d: components.CompliantMotion,
    K_star: np.ndarray,
    p_idx: int = 0,
    validated_samples=[],
) -> components.CompliantMotion:
    U0 = refine_p(b.particles[p_idx], CF_d, K_star)
    U0 = U0 + validated_samples
    print(f"{len(U0)=}")
    if len(U0) == 0:
        return None, float("-inf"), False
    P1_next = dynamics.f_cspace(b.particles[int(not p_idx)], U0)
    U = []
    success = True
    relaxed_CF_d = relax_CF(CF_d)
    for i, p1_next in enumerate(P1_next):
        if p1_next.satisfies_contact(relaxed_CF_d):
            U.append(U0[i])
    if len(U) == 0:
        print("no intersect")
        success = False
        U = U0
    best_u = None
    most_certainty = float("-inf")
    for u in U:
        posterior = dynamics.f_bel(b, u)
        certainty = len(posterior.contact_state())
        if certainty > most_certainty:
            most_certainty = certainty
            best_u = u
    print(f"{most_certainty=}")
    return best_u, most_certainty, success


def refine_b(
    b: state.Belief, CF_d: components.ContactState
) -> components.CompliantMotion:
    print(f"{CF_d=}")
    K_star, samples = solve_for_compliance(b.particles[0], CF_d)
    best_u_0, certainty_0, success = score_tree_root(
        b, CF_d, K_star, p_idx=0, validated_samples=samples
    )
    if success:
        return best_u_0
    best_u_1, certainty_1, success = score_tree_root(b, CF_d, K_star, p_idx=1)
    if success:
        return best_u_1
    if best_u_0 is None and best_u_1 is None:
        return None
    assert certainty_0 >= 0 or certainty_1 >= 0
    if certainty_0 >= certainty_1:
        return best_u_0
    else:
        return best_u_1
