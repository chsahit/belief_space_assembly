from typing import List

import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
from simulation import generate_contact_set, ik_solver


def evaluate_K(
    p: state.Particle,
    CF_d: components.ContactState,
    K: np.ndarray,
    targets: List[RigidTransform] = None,
) -> List[components.CompliantMotion]:
    scores = []
    negative_motions = []
    nominal = p.X_WG
    if targets is None:
        targets = generate_contact_set.project_manipuland_to_contacts(
            p, CF_d, num_samples=refinement_samples
        )

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


def K_t_opt(p: state.Particle) -> np.ndarray:
    pass


def solve_for_compliance(
    p: state.Particle, CF_d: components.ContactState
) -> np.ndarray:
    targets = generate_contact_set.project_manipuland_to_contacts(
        p, CF_d, num_samples=16
    )
    K_opt = K_t_opt(p)
    validated_samples, _ = evaluate_K(p, CF_d, K_opt, targets=targets)
    succ_count = len(validated_samples)
    for i in range(3):
        K_curr = np.copy(K_init)
        K[i, i] = components.soft[i]
        curr_samples, _ = evaluate_K(p, CF_d, K_curr, targets=targets)
        curr_succ_count = len(curr_samples)
        if curr_succ_count == len(targets):
            return K_curr, curr_validated_samples
        if curr_succ_count > succ_count:
            succ_count = curr_succ_count
            validated_samples = curr_samples
            K_opt = K_curr
    return K_opt, validated_samples


def solve_for_compliance(b: state.Belief) -> np.ndarray:
    # idea, average compliance across particles?
    pass
