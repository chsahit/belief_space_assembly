from typing import List, Tuple

import numpy as np
from pydrake.all import MathematicalProgram, RigidTransform, Solve

import components
import dynamics
import state
import utils
from simulation import generate_contact_set, ik_solver

np.set_printoptions(precision=5, suppress=True)


def evaluate_K(
    p: state.Particle,
    CF_d: components.ContactState,
    K: np.ndarray,
    targets: List[RigidTransform] = None,
) -> List[components.CompliantMotion]:
    scores = []
    negative_motions = []
    if targets is None:
        targets = generate_contact_set.project_manipuland_to_contacts(
            p, CF_d, num_samples=32
        )

    X_GC = RigidTransform([0, 0, 0.0])
    targets = [target.multiply(X_GC) for target in targets]
    # visualize.visualize_targets(p, targets)
    motions = [components.CompliantMotion(X_GC, target, K) for target in targets]
    motions = [ik_solver.update_motion_qd(m) for m in motions]
    # for m in motions:
    #     dynamics.simulate(p, m, vis=True)
    # breakpoint()
    P_next = dynamics.f_cspace(p, motions)
    U = []
    for i, p_next in enumerate(P_next):
        if p_next.satisfies_contact(CF_d):
            U.append(motions[i])
        else:
            negative_motions.append(motions[i])
            scores.append(0)
    if p.satisfies_contact(CF_d) and len(U) == 0:
        breakpoint()
    return U, (negative_motions, scores)


def is_psd(A: np.ndarray) -> bool:
    Ap = A + 1e-3 * np.eye(3)
    if np.allclose(Ap, Ap.T):
        try:
            np.linalg.cholesky(Ap)
            return True
        except np.linalg.LinAlgError:
            return False
    return False


def compute_normal(pt, cspace, candidate_normals, step_size) -> np.ndarray:
    best_normal = None
    min_dist = float("inf")
    for n in candidate_normals:
        candidate_pt = pt + step_size * np.array(n)
        if cspace.PointInSet(candidate_pt):
            continue
        prog = MathematicalProgram()
        closest_pt = prog.NewContinuousVariables(3, "cpt")
        prog.SetInitialGuess(closest_pt, pt)
        dist_var = (candidate_pt - closest_pt).dot(candidate_pt - closest_pt)
        prog.AddCost(dist_var)
        cspace.AddPointInSetConstraints(prog, closest_pt)
        pt_star = Solve(prog).GetSolution(closest_pt)
        dist = np.linalg.norm(pt_star - pt)
        if dist < min_dist:
            min_dist = dist
            best_normal = n
    return best_normal


def K_r_opt(p: state.Particle) -> np.ndarray:
    if len(p.contacts) == 0:
        return components.stiff[:3]
    K = list()
    delta = 0.05
    for i in range(3):
        rpy = np.array([0, 0, 0])
        rpy[i] = delta
        delta_pos = utils.xyz_rpy_deg([0, 0, 0], rpy)
        delta_neg = utils.xyz_rpy_deg([0, 0, 0], -rpy)
        cspace_pos = generate_contact_set.make_cspace(p, p.contacts, delta_pos)
        cspace_neg = generate_contact_set.make_cspace(p, p.contacts, delta_neg)
        if cspace_pos.PointInSet(p.X_WM.translation()) and cspace_neg.PointInSet(
            p.X_WM.translation()
        ):
            K.append(components.soft[i])
        else:
            K.append(components.stiff[i])

    return np.array(K)


def K_t_opt(p: state.Particle) -> Tuple[np.ndarray, np.ndarray]:
    K = np.diag(components.stiff[3:])
    if len(p.contacts) == 0:
        return K, np.array([0, 0, 0])
    K[2, 2] = components.soft[5]
    cspace = generate_contact_set.make_cspace(p, p.contacts)
    pt = p.X_WM.translation()
    normals = list()
    dirs = {0: -1, 1: -0.5, 2: 0, 3: 0.5, 4: 1}
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 2 and y == 2 and z == 2:
                    continue
                normals.append([dirs[x], dirs[y], dirs[z]])

    for i in range(3):
        best_normal = compute_normal(pt, cspace, normals, 1e-4 * (10**i))
        if best_normal is not None:
            break
    if best_normal is None:
        print("warning: could not id normal vector")
        return K, None
    best_normal = np.abs(best_normal / np.linalg.norm(best_normal))
    I3 = np.eye(3)
    vecs = np.concatenate((np.array([best_normal]), I3))
    q, r = np.linalg.qr(vecs.T)
    q[:, 0] = best_normal
    basis_vectors = q[:, [1, 2, 0]]
    opt = basis_vectors @ K @ np.linalg.inv(basis_vectors)
    if np.linalg.norm(best_normal, ord=1) > 1.01 and False:
        print(f"{best_normal=}")
        print(f"basis_vectors=\n{basis_vectors}")
        print(f"opt=\n{opt}")
    # assert is_psd(opt)
    return opt, best_normal


def solve_for_compliance(
    p: state.Particle, CF_d: components.ContactState
) -> np.ndarray:
    del CF_d
    K_r = np.diag(K_r_opt(p))
    K_t, _ = K_t_opt(p)
    K_opt = np.zeros((6, 6))
    K_opt[:3, :3] = K_r
    K_opt[3:, 3:] = K_t
    print(f"K_opt diagonal={np.diag(K_opt)}")
    return K_opt, []


def hybrid_solve_for_compliance(
    p: state.Particle, CF_d: components.ContactState
) -> np.ndarray:
    targets = generate_contact_set.project_manipuland_to_contacts(
        p, CF_d, num_samples=16
    )
    # if "top" in str(CF_d) and len(targets) == 16:
    #     visualize.visualize_targets(p, targets)
    K_opt_3, best_normal = K_t_opt(p)
    K_opt = np.diag(components.stiff)
    K_opt[3:, 3:] = K_opt_3
    validated_samples, _ = evaluate_K(p, CF_d, K_opt, targets=targets)
    succ_count = len(validated_samples)
    print(f"{best_normal=}, {succ_count=}")
    for i in range(3):
        K_curr = np.copy(K_opt)
        K_curr[i, i] = components.soft[i]
        curr_samples, _ = evaluate_K(p, CF_d, K_curr, targets=targets)
        curr_succ_count = len(curr_samples)
        if curr_succ_count == len(targets):
            return K_curr, curr_samples
        if curr_succ_count > succ_count:
            succ_count = curr_succ_count
            print(f"setting index {i} compliant, {succ_count=}")
            validated_samples = curr_samples
            K_opt = K_curr
    print(f"K_opt=\n{K_opt}")
    return K_opt, validated_samples
