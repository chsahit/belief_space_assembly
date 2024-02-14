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
    VPolytope,
)

import components
import mr
import state
import utils
from simulation import hyperrectangle, ik_solver

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


def relax_CF(CF_d: components.ContactState) -> components.ContactState:
    relaxed_CF_d = set()
    for env_contact, manip_contact in CF_d:
        e_u = env_contact.rfind("_")
        r_ec = env_contact[:e_u]
        e_m = manip_contact.rfind("_")
        r_mc = manip_contact[:e_m]
        relaxed_CF_d.add((r_ec, r_mc))
    return relaxed_CF_d


def tf(X_MMt, p_MV):
    homogenous = np.array([p_MV[0], p_MV[1], p_MV[2], 1])
    homogenous_tf = X_MMt.inverse().GetAsMatrix4() @ homogenous
    r3 = np.array([homogenous_tf[0], homogenous_tf[1], homogenous_tf[2]])
    return r3


def tf_HPolyhedron(H: HPolyhedron, X: RigidTransform) -> HPolyhedron:
    scale_factor = 100000
    H_big = H.Scale(scale_factor)
    vrep = VPolytope(H_big)
    verts = vrep.vertices()
    assert verts.shape[1] == 8
    transformed_verts = np.array([tf(X, vert) for vert in verts.T])
    transformed_vrep = VPolytope(transformed_verts.T)
    transformed_hrep = HPolyhedron(transformed_vrep).Scale(1.0 / scale_factor)
    return transformed_hrep


def generate_noised(p: state.Particle, X_WM, CF_d, verbose=False):
    constraints = p.constraints
    relaxed_CF_d = relax_CF(CF_d)
    # r_vel = gen.uniform(low=-0.05, high=0.05, size=3)
    r_vel = gen.uniform(low=-0.00, high=0.00, size=3)
    t_vel = gen.uniform(low=-0.01, high=0.01, size=3)
    random_vel = np.concatenate((r_vel, t_vel))
    X_MMt = RigidTransform(mr.MatrixExp6(mr.VecTose3(random_vel)))
    X_WMt = X_WM.multiply(X_MMt)
    contact_manifold = make_cspace(p, relaxed_CF_d, tf=RigidTransform(X_WMt.rotation()))
    if contact_manifold.PointInSet(X_WMt.translation()):
        if verbose:
            print(f"(point as is) {X_WMt.translation()=}")
        return X_WMt, X_MMt
    else:
        if verbose:
            print(f"off manifold, sample_translated={X_WM.translation()}")
        return X_WM, RigidTransform()


def make_cspace(
    p: state.Particle,
    CF_d: components.ContactState,
    tf: RigidTransform = RigidTransform(),
) -> Intersection:
    contact_manifold = None
    constraints = p.constraints
    for env_poly, manip_poly_name in CF_d:
        A_env, b_env = constraints[env_poly]
        env_geometry = HPolyhedron(A_env, b_env)
        A_manip, b_manip = p._manip_poly[manip_poly_name]
        rotatated_manip = tf_HPolyhedron(HPolyhedron(A_manip, b_manip), tf)
        reflection = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0]]
        )
        manip_geometry = tf_HPolyhedron(rotatated_manip, RigidTransform(reflection))
        minkowski_sum = MinkowskiSum(env_geometry, manip_geometry)
        if contact_manifold is None:
            contact_manifold = minkowski_sum
        else:
            contact_manifold = Intersection(contact_manifold, minkowski_sum)
    assert not contact_manifold.IsEmpty()
    return contact_manifold


def compute_samples_from_contact_set(
    p: state.Particle, CF_d: components.ContactState, num_samples: int = 1
) -> List[np.ndarray]:
    contact_manifold = make_cspace(p, CF_d)
    samples = []
    cm_hyper_rect, bounds = hyperrectangle.CalcAxisAlignedBoundingBox(contact_manifold)
    interior_pts = rejection_sample(contact_manifold, bounds, num_samples=num_samples)
    for interior_pt in interior_pts:
        is_interior = True
        random_direction = gen.uniform(low=-1, high=1, size=3)
        random_direction = random_direction / np.linalg.norm(random_direction)
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
    p_WMs = compute_samples_from_contact_set(p, CF_d, num_samples=num_samples)
    X_WMs = [RigidTransform(p_WM) for p_WM in p_WMs]
    verbose = (num_samples == 16) and ("top" in str(CF_d))
    verbose = False
    samples_noised = [generate_noised(p, X_WM, CF_d, verbose=verbose) for X_WM in X_WMs]
    for (X_WMt, X_MMt) in samples_noised:
        X_WG = X_WMt.multiply(p.X_GM.inverse())
        q_r = ik_solver.gripper_to_joint_states(X_WG)
        new_p = p.deepcopy()
        new_p.q_r = q_r
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
