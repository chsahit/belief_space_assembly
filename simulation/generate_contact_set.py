from typing import List

import numpy as np
from pydrake.all import HPolyhedron, Intersection, MinkowskiSum

import components
import state
from simulation import annotate_geoms

gen = np.random.default_rng(0)


def compute_samples_from_contact_set(
    p: state.Particle, CF_d: components.ContactState, num_samples: int = 1
) -> List[np.ndarray]:
    contact_manifold = None
    constraints = p.constraints
    samples = []
    for env_poly, _ in CF_d:
        A_env, b_env = constraints[env_poly]
        env_geometry = HPolyhedron(A_env, b_env)
        A_manip, b_manip = p._manip_poly
        A_manip = -1 * A_manip
        manip_geometry = HPolyhedron(A_manip, b_manip)
        minkowski_sum = MinkowskiSum(env_geometry, manip_geometry)
        if contact_manifold is None:
            contact_manifold = minkowski_sum
        else:
            contact_manifold = Intersection(contact_manifold, minkowski_sum)
    assert not contact_manifold.IsEmpty()
    for sample_id in range(num_samples):
        interior_pt = contact_manifold.MaybeGetFeasiblePoint()
        is_interior = True
        random_direction = gen.uniform(low=-1, high=1, size=3)
        step_size = 1e-4
        while not is_interior:
            interior_pt += epsilon * random_direction
            is_interior = contact_manifold.PointInSet(interior_pt)
        interior_pt -= epsilon * random_direction
        samples.append(interior_pt)
    return samples
