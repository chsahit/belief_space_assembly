from typing import List

import numpy as np
from pydrake.all import (
    ConvexSet,
    HPolyhedron,
    Hyperrectangle,
    Intersection,
    MinkowskiSum,
    RandomGenerator,
    RigidTransform,
    RotationMatrix,
    Simulator,
)

import components
import state
from simulation import annotate_geoms, ik_solver

gen = np.random.default_rng(1)
drake_rng = RandomGenerator()


def sample_from_cvx_set(
    cvx_set: ConvexSet, hyper_rect: Hyperrectangle = None, max_samples=100
) -> np.ndarray:
    if hyper_rect is None:
        hyper_rect = Hyperrectangle.MaybeCalcAxisAlignedBoundingBox()
    samples = 0
    while samples < max_samples:
        rect_sample = hyper_rect.UniformSample(drake_rng)
        if cvx_set.PointInSet(rect_sample):
            return rect_sample

    return cvx_set.MaybeGetFeasiblePoint()


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
    cm_hyper_rect = Hyperrectangle.MaybeCalcAxisAlignedBoundingBox(contact_manifold)
    for sample_id in range(num_samples):
        interior_pt = sample_from_cvx_set(contact_manifold, cm_hyper_rect)
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
    print(f"{CF_d=}")
    while len(projections) < num_samples:
        sample = compute_samples_from_contact_set(p, CF_d)[0]
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
    offset = RigidTransform([0, 0, 0.005])
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
