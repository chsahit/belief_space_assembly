from typing import List

import numpy as np
from pydrake.all import RigidTransform

import components
import mr
from belief import belief_state, dynamics
from simulation import ik_solver


def find_nearest_valid_target(
    p: belief_state.Particle, CF_d: components.ContactState
) -> RigidTransform:
    return ik_solver.project_manipuland_to_contacts(p, CF_d)


def e(i):
    v = np.zeros((6,))
    v[i] = 1
    return v


def _compute_displacements(density: int) -> List[np.ndarray]:
    displacements = [np.zeros((6,))]
    for i in range(6):
        if i < 3:
            Delta = np.linspace(-0.2, 0.2, density).tolist()
        else:
            Delta = np.linspace(-0.02, 0.02, density).tolist()
        displacement_i = [delta * e(i) for delta in Delta]
        displacements.extend(displacement_i)
    return displacements


def grow_motion_set(
    X_GC: RigidTransform,
    K: np.ndarray,
    CF_d: components.ContactState,
    p: belief_state.Particle,
    density: int = 5,
) -> List[components.CompliantMotion]:

    U = []
    X_WGd_0 = find_nearest_valid_target(p, CF_d).multiply(X_GC)
    X_WCd_0 = X_WGd_0.multiply(X_GC)
    U_candidates = []
    for displacement in _compute_displacements(density):
        X_WCd_i = X_WCd_0.multiply(
            RigidTransform(mr.MatrixExp6(mr.VecTose3(displacement)))
        )
        u_i = components.CompliantMotion(X_GC, X_WCd_i, K)
        U_candidates.append(u_i)

    P_results = dynamics.f_cspace(p, U_candidates)
    for idx, p in enumerate(P_results):
        if p.satisfies_contact(CF_d):
            U.append(U_candidates[idx])
    return U


def intersect_motion_set(
    X_GC: RigidTransform,
    K: np.ndarray,
    b: belief_state.Belief,
    CF_d: components.ContactState,
) -> components.CompliantMotion:
    pass
