# IK solving is bad, this exists now
import os
from typing import List

import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import mr
import state
from simulation import ik_solver

gen = np.random.default_rng(0)

if os.uname()[1] == "londonsystem":
    compliance_samples = 16
    refinement_samples = 32
else:
    compliance_samples = 500
    refinement_samples = 1500
print(f"{compliance_samples=}, {refinement_samples=}")


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
    X_WG = p.X_WG
    targets = generate_targets(
        p.X_WG, r_bound=0.1, t_bound=0.03, count=compliance_samples
    )
    K_opt = np.copy(components.stiff)
    succ_count = len(refine_p(p, CF_d, K_opt, targets=targets))
    print(f"{K_opt=}, {succ_count=}")
    for i in range(6):
        K_curr = np.copy(K_opt)
        K_curr[i] = components.soft[i]
        curr_succ_count = len(refine_p(p, CF_d, K_curr, targets=targets))
        print(f"{K_curr=}, {curr_succ_count=}")
        if curr_succ_count > succ_count:
            succ_count = curr_succ_count
            K_opt = K_curr

    K_opt_soft = np.copy(components.soft)
    succ_count_soft = len(refine_p(p, CF_d, K_opt_soft, targets=targets))
    print(f"{K_opt_soft=}, {succ_count_soft=}")
    for i in range(6):
        K_curr = np.copy(K_opt_soft)
        K_curr[i] = components.stiff[i]
        curr_succ_count_soft = len(refine_p(p, CF_d, K_curr, targets=targets))
        print(f"{K_curr=}, {curr_succ_count_soft=}")
        if curr_succ_count_soft > succ_count_soft:
            succ_count_soft = curr_succ_count_soft
            K_opt_soft = K_curr
    if succ_count_soft > succ_count:
        K_opt = K_opt_soft
    print(f"{K_opt=}")
    return K_opt


def refine_p(
    p: state.Particle,
    CF_d: components.ContactState,
    K: np.ndarray,
    targets: List[RigidTransform] = None,
) -> List[components.CompliantMotion]:
    nominal = p.X_WG
    # nominal = ik_solver.project_manipuland_to_contacts(p, CF_d)
    if targets is None:
        targets = generate_targets(
            nominal, r_bound=0.1, t_bound=0.03, count=refinement_samples
        )
    X_GC = RigidTransform()
    motions = [components.CompliantMotion(X_GC, target, K) for target in targets]
    P_next = dynamics.f_cspace(p, motions)
    U = []
    for i, p_next in enumerate(P_next):
        if p_next.satisfies_contact(CF_d):
            U.append(motions[i])
    return U


def refine_b(
    b: state.Belief, CF_d: components.CompliantMotion
) -> components.CompliantMotion:
    K_star = solve_for_compliance(b.particles[0], CF_d)
    U0 = refine_p(b.particles[0], CF_d, K_star)
    print(f"{len(U0)=}")
    if len(U0) > 0:
        P1_next = dynamics.f_cspace(b.particles[1], U0)
    else:
        P1_next = []
    for i, p1_next in enumerate(P1_next):
        if p1_next.satisfies_contact(CF_d):
            return U0[i]
    U1 = refine_p(b.particles[1], CF_d, K_star)
    print(f"{len(U1)=}")
    if len(U1) > 0:
        P0_next = dynamics.f_cspace(b.particles[0], U1)
    else:
        P0_next = []
    for i, p0_next in enumerate(P0_next):
        if p0_next.satisfies_contact(CF_d):
            return U1[i]
    print("no motion found")
    return None
