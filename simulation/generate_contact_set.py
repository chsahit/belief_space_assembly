import pickle
import random
from typing import List

import cdd
import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import ConvexSet, HPolyhedron, RandomGenerator, RigidTransform

import components
import cspace
import mr
import state
from simulation import hyperrectangle, ik_solver

random.seed(0)
gen = np.random.default_rng(1)
drake_rng = RandomGenerator()


def rejection_sample(cvx_set: ConvexSet, bounds, num_samples=1):
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


scale_factor = 100000


def Scale(H: HPolyhedron, scale: float = 100) -> HPolyhedron:
    b = H.b()
    b *= scale
    return HPolyhedron(H.A(), b)


def MatToArr(m: cdd.Matrix) -> np.ndarray:
    return np.array([m[i] for i in range(m.row_size)])


def HPolyhedronToVRepr(H: HPolyhedron) -> cdd.Matrix:
    A, b = (H.A(), H.b())
    H_repr = np.hstack((np.array([b]).T, -A))
    mat = cdd.Matrix(H_repr, number_type="float")
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    return poly.get_generators()


def VerticesToHPolyhedron(vertices: np.ndarray) -> HPolyhedron:
    V_repr_tf = np.hstack((np.ones((vertices.shape[0], 1)), vertices))
    try:
        mat_tf_V = cdd.Matrix(V_repr_tf, number_type="float")
        mat_tf_V.rep_type = cdd.RepType.GENERATOR
        poly_tf = cdd.Polyhedron(mat_tf_V)
    except:
        V_repr_tf = np.round(V_repr_tf, decimals=8)
        V_repr_tf += 1e-6
        mat_tf_V.rep_type = cdd.RepType.GENERATOR
        poly_tf = cdd.Polyhedron(mat_tf_V)
    H_repr_tf = MatToArr(poly_tf.get_inequalities())
    b_tf = H_repr_tf[:, 0]
    A_tf = -1 * H_repr_tf[:, 1:]
    return HPolyhedron(A_tf, b_tf)


def reflect_HPolyhedron(H: HPolyhedron) -> HPolyhedron:
    V_repr = MatToArr(HPolyhedronToVRepr(H))
    if V_repr.shape[0] < 8:
        breakpoint()
    vertices = V_repr[:, 1:]
    vertices_transformed = []
    for v_idx in range(vertices.shape[0]):
        vertices_transformed.append(-1 * vertices[v_idx])
    vertices_transformed = np.array(vertices_transformed)
    return VerticesToHPolyhedron(vertices_transformed)


def tf_HPolyhedron(H: HPolyhedron, X: RigidTransform) -> HPolyhedron:
    V_repr = MatToArr(HPolyhedronToVRepr(H))
    vertices = V_repr[:, 1:]
    vertices_transformed = []
    for v_idx in range(vertices.shape[0]):
        vertices_transformed.append(tf(X, vertices[v_idx]))
    vertices_transformed = np.array(vertices_transformed)
    return VerticesToHPolyhedron(vertices_transformed)


def lowest_pt(X_WMt: RigidTransform, H: HPolyhedron) -> np.ndarray:
    from pydrake.all import MathematicalProgram, Solve

    eqns = tf_HPolyhedron(H, X_WMt.inverse())
    mp = MathematicalProgram()
    opt = mp.NewContinuousVariables(3, "opt")
    mp.AddCost(opt[2])
    eqns.AddPointInSetConstraints(mp, opt)
    return Solve(mp).GetSolution(opt)


def generate_noised(p: state.Particle, X_WM, CF_d, verbose=False):
    relaxed_CF_d = relax_CF(CF_d)
    r_vel = gen.uniform(low=-0.05, high=0.05, size=3)
    # r_vel = gen.uniform(low=-0.00, high=0.00, size=3)
    t_vel = gen.uniform(low=-0.01, high=0.01, size=3)
    random_vel = np.concatenate((r_vel, t_vel))
    X_MMt = RigidTransform(mr.MatrixExp6(mr.VecTose3(random_vel)))
    X_WMt = X_WM.multiply(X_MMt)
    contact_manifold = make_cspace(p, relaxed_CF_d, tf=RigidTransform(X_WMt.rotation()))
    if contact_manifold.PointInSet(X_WMt.translation()):
        if verbose:
            print(f"(point as is) {X_WMt.translation()=}")
        return X_WMt, X_MMt
    elif contact_manifold.PointInSet(X_WM.translation()):
        direction = X_WMt.translation() - X_WM.translation()
        pt = np.copy(X_WM.translation())
        while contact_manifold.PointInSet(pt):
            pt += 1e-4 * direction
        pt -= 1e-4 * direction
        X_WMt_new = RigidTransform(X_WMt.rotation(), pt)
        X_MMt_new = X_WM.inverse().multiply(X_WMt_new)
        """
        if pt[2] > 0.081 and verbose:
            hpol = HPolyhedron(*p._manip_poly["block::Box"])
            breakpoint()
            print(f"{lowest_pt(X_WMt_new, hpol)=}")
        """
        if verbose:
            print(f"computed: {X_WMt_new.translation()=}")
            hpol = HPolyhedron(*p._manip_poly["block::Box"])
            print(f"{lowest_pt(X_WMt_new, hpol)=}")
        return X_WMt_new, X_MMt_new
    else:
        if verbose:
            print(f"off manifold, sample_translated={X_WM.translation()}")
        return X_WM, RigidTransform()


def make_cspace(
    p: state.Particle,
    CF_d: components.ContactState,
    tf: RigidTransform = RigidTransform(),
) -> HPolyhedron:
    constraints = p.constraints
    p_cspace = None
    for env_poly, manip_poly_name in CF_d:
        env_geometry = HPolyhedron(*constraints[env_poly])
        manip_geometry = HPolyhedron(*p._manip_poly[manip_poly_name])
        rotated_manip = cspace.TF_HPolyhedron(manip_geometry, tf)
        msum = cspace.minkowski_sum("", env_geometry, "", rotated_manip).geometry
        if p_cspace is None:
            p_cspace = msum
        else:
            p_cspace = p_cspace.Intersection(p_cspace)
    assert not p_cspace.IsEmpty()
    return p_cspace


def compute_samples_from_contact_set(
    p: state.Particle, CF_d: components.ContactState, num_samples: int = 1
) -> List[np.ndarray]:
    contact_manifold = make_cspace(p, CF_d)
    if len(CF_d) == 1:
        samples = sample_from_contact_triangle(p, CF_d, num_samples=num_samples)
        return samples
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


def sample_from_contact_triangle(
    p: state.Particle, CF_d: components.ContactState, num_samples: int = 1
) -> List[np.ndarray]:
    triangles = None
    samples = []
    with open("cspace_surface.pkl", "rb") as f:
        triangles = pickle.load(f)
    assert triangles is not None
    triangles_cf = triangles[CF_d]
    if len(triangles_cf) == 0:
        breakpoint()
    sample_attempts = 0
    while True:
        sample_attempts += 1
        verts = random.choice(triangles_cf)
        w = gen.uniform(low=0, high=1, size=3)
        w /= np.sum(w)
        pt = w[0] * verts[0] + w[1] * verts[1] + w[2] * verts[2]
        if pt[1] > 0.035 or pt[0] > 0.57 or pt[1] < -0.035:
            continue
        samples.append(pt)
        if len(samples) == num_samples:
            break
    return samples


def project_manipuland_to_contacts(
    p: state.Particle, CF_d: components.ContactState, num_samples: int = 1
) -> List[RigidTransform]:
    projections = []
    p_WMs = compute_samples_from_contact_set(p, CF_d, num_samples=num_samples)
    X_WMs = [RigidTransform(p_WM) for p_WM in p_WMs]
    verbose = (num_samples == 16) and ("top" in str(CF_d))
    verbose = False
    samples_noised = [generate_noised(p, X_WM, CF_d, verbose=verbose) for X_WM in X_WMs]
    samples_noised = [(X_WM, None) for X_WM in X_WMs]
    for X_WMt, X_MMt in samples_noised:
        X_WG = X_WMt.multiply(p.X_GM.inverse())
        q_r = ik_solver.gripper_to_joint_states(X_WG)
        new_p = p.deepcopy()
        new_p.q_r = q_r
        projections.append(X_WG)
    return projections
