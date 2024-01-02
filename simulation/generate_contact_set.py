import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import (
    ConvexSet,
    HPolyhedron,
    Intersection,
    MinkowskiSum,
    RandomGenerator,
    RigidTransform,
    RotationMatrix,
    Simulator,
)

import components
import state
from simulation import annotate_geoms, hyperrectangle, ik_solver

random.seed(0)
gen = np.random.default_rng(1)
drake_rng = RandomGenerator()


def rejection_sample(cvx_set: ConvexSet, bounds, num_samples=1):
    mixing_amount = 400  # can just be num samples
    valid_samples = []
    while len(valid_samples) < num_samples:
        sample = gen.uniform(bounds[0], bounds[1])
        if cvx_set.PointInSet(sample):
            # plt.scatter(sample[0], sample[1])
            valid_samples.append(sample)
    # plt.show()
    return valid_samples[:num_samples]


def hit_and_run_sample(
    cvx_set: ConvexSet, hyper_rect: HPolyhedron, num_samples: int = 1
) -> np.ndarray:
    rect_sample = hyper_rect.UniformSample(drake_rng)
    mixing_amount = 400
    valid_samples = []
    while len(valid_samples) < mixing_amount:
        rect_sample = hyper_rect.UniformSample(drake_rng, rect_sample)
        plt.scatter(rect_sample[0], rect_sample[1])
        if cvx_set.PointInSet(rect_sample):
            valid_samples.append(rect_sample)
    subsamples = []
    for i in range(num_samples):
        subsamples.append(random.choice(valid_samples))
    plt.show()
    return subsamples


def compute_samples_from_contact_set(
    p: state.Particle, CF_d: components.ContactState, num_samples: int = 1
) -> List[np.ndarray]:
    contact_manifold = None
    constraints = p.constraints
    samples = []
    for env_poly, manip_poly_name in CF_d:
        A_env, b_env = constraints[env_poly]
        env_geometry = HPolyhedron(A_env, b_env)
        A_manip, b_manip = p._manip_poly[manip_poly_name]
        # breakpoint()
        A_manip = -1 * A_manip
        manip_geometry = HPolyhedron(A_manip, b_manip)
        minkowski_sum = MinkowskiSum(env_geometry, manip_geometry)
        if contact_manifold is None:
            contact_manifold = minkowski_sum
        else:
            contact_manifold = Intersection(contact_manifold, minkowski_sum)
    assert not contact_manifold.IsEmpty()
    cm_hyper_rect, bounds = hyperrectangle.CalcAxisAlignedBoundingBox(contact_manifold)
    # interior_pts = hit_and_run_sample(contact_manifold, cm_hyper_rect, num_samples=num_samples)
    interior_pts = rejection_sample(contact_manifold, bounds, num_samples=num_samples)
    for interior_pt in interior_pts:
        is_interior = True
        random_direction = gen.uniform(low=-1, high=1, size=3)
        # random_direction = np.array([1.0, -0.0, 0.0])
        # random_direction[0] = abs(random_direction[0])
        random_direction = random_direction / np.linalg.norm(random_direction)
        # print(f"{random_direction=}")
        step_size = 5e-5
        while is_interior:
            interior_pt += step_size * random_direction
            is_interior = contact_manifold.PointInSet(interior_pt)
        interior_pt -= step_size * random_direction
        assert contact_manifold.PointInSet(interior_pt)
        samples.append(interior_pt)
    return samples


def _project_manipuland_to_contacts(
    p: state.Particle, CF_d: components.ContactState, num_samples: int = 1
) -> List[RigidTransform]:
    projections = []
    samples = compute_samples_from_contact_set(p, CF_d, num_samples=num_samples)
    for sample in samples:
        projection = RigidTransform(RotationMatrix(), sample)
        X_WG = projection.multiply(p.X_GM.inverse())
        q_r = ik_solver.gripper_to_joint_states(X_WG)
        new_p = p.deepcopy()
        new_p.q_r = q_r
        # depth = collision_depth(new_p)
        projections.append(X_WG)

    return projections


def project_manipuland_to_contacts(
    p: state.Particle, CF_d: components.ContactState, num_samples: int = 1
) -> List[RigidTransform]:
    offset = RigidTransform([0, 0, 0.000])
    projections_pre = _project_manipuland_to_contacts(p, CF_d, num_samples=num_samples)
    projections = [p.multiply(offset) for p in projections_pre]
    return projections


def collision_depth(p: state.Particle) -> float:
    diagram, _ = p.make_plant(vis=False)
    simulator = Simulator(diagram)
    simulator.AdvanceTo(0.1)
    worst_collision_amt = float("inf")
    wc = None
    for k, v in p.sdf.items():
        if v < worst_collision_amt:
            worst_collision_amt = v
            wc = k
    return worst_collision_amt
