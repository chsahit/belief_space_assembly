from typing import List

import numpy as np
from pydrake.all import HPolyhedron, Intersection, MinkowskiSum, RigidTransform

import components
import state
from simulation import annotate_geoms

gen = np.random.default_rng(1)


def compute_samples_from_contact_set(
    p: state.Particle, CF_d: components.ContactState, num_samples: int = 1
) -> List[np.ndarray]:
    contact_manifold = None
    constraints = p.constraints
    samples = []
    print(f"{CF_d=}")
    for env_poly, manip_poly_name in CF_d:
        A_env, b_env = constraints[env_poly]
        env_geometry = HPolyhedron(A_env, b_env)
        A_manip, b_manip = p._manip_poly[manip_poly_name]
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
        # random_direction = np.array([-1.0, 0.0, 1.0])
        # random_direction[-1] = abs(random_direction[-1])
        random_direction = random_direction / np.linalg.norm(random_direction)
        print(f"{random_direction=}")
        step_size = 1e-4
        while is_interior:
            interior_pt += step_size * random_direction
            is_interior = contact_manifold.PointInSet(interior_pt)
        interior_pt -= step_size * random_direction
        assert contact_manifold.PointInSet(interior_pt)
        samples.append(interior_pt)
    return samples


def project_manipuland_to_contacts(
    p: state.Particle, CF_d: components.ContactState, num_samples: int = 1
) -> List[RigidTransform]:
    samples = compute_samples_from_contact_set(p, CF_d, num_samples=num_samples)
    projections = [RigidTransform(p.X_WM.rotation(), sample) for sample in samples]
    projections = [projection.multiply(p.X_GM.inverse()) for projection in projections]
    return projections
