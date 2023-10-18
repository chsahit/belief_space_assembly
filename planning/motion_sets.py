import random
from typing import List

import numpy as np
from pydrake.all import (
    HPolyhedron,
    RandomGenerator,
    RigidTransform,
    RotationMatrix,
    VPolytope,
)
from sklearn.decomposition import PCA

import components
import dynamics
import mr
import state
import utils
import visualize
from simulation import ik_solver


def find_nearest_valid_target(
    p: state.Particle, CF_d: components.ContactState
) -> RigidTransform:
    return ik_solver.project_manipuland_to_contacts(p, CF_d)


def e(i):
    v = np.zeros((6,))
    v[i] = 1
    return v


def _compute_displacements(density: int) -> List[np.ndarray]:
    displacements = [np.zeros((6,))]
    for i in range(6):
        if i == 0:
            continue
        if i < 3:
            Delta = np.linspace(-0.25, 0.25, density).tolist()
        else:
            Delta = np.linspace(-0.03, 0.03, density).tolist()
        if density % 2 != 0:
            middle = int((density - 1) / 2)
            Delta = np.concatenate((Delta[:middle], Delta[middle + 1 :]))
        displacement_i = [delta * e(i) for delta in Delta]
        displacements.extend(displacement_i)
    return displacements


def logmap_setpoints(X_WCs_batch: List[List[RigidTransform]]) -> List[List[np.ndarray]]:
    assert len(X_WCs_batch) > 1
    nominal = X_WCs_batch[0][0].rotation()
    logmap_batch = []
    for X_WCs in X_WCs_batch:
        logmapped = []
        for X_WC in X_WCs:
            t = X_WC.translation()
            R_WC = X_WC.rotation()
            delta = R_WC.multiply(nominal.inverse()).matrix()
            # print("delta:", utils.rt_to_str(RigidTransform(RotationMatrix(delta), t)))
            r = mr.so3ToVec(mr.MatrixLog3(delta))
            logmapped.append(np.concatenate((r, t)))
        logmap_batch.append(logmapped)
    return logmap_batch


def expmap_intersection(sp: np.ndarray, origin: RigidTransform) -> RigidTransform:
    R_bar = mr.MatrixExp3(mr.VecToso3(sp[:3]))
    R = RotationMatrix(R_bar).multiply(origin.rotation())
    return RigidTransform(R, sp[3:])


def sample_to_motion(
    sp: np.ndarray, origin: RigidTransform, mapping, u_nom
) -> components.CompliantMotion:
    X_WCd = expmap_intersection(mapping.inverse_transform([sp][0]), origin)
    u = components.CompliantMotion(u_nom.X_GC, X_WCd, u_nom.K)
    return u


def grow_motion_set(
    X_GC: RigidTransform,
    K: np.ndarray,
    CF_d: components.ContactState,
    p: state.Particle,
    density: int = 9,
    vis=False,
) -> List[components.CompliantMotion]:

    U = []
    X_WGd_0 = find_nearest_valid_target(p, CF_d)
    if X_WGd_0 is None:
        print("IK solve failed, returning none")
        return U
    print("X_WGd_0: ", utils.rt_to_str(X_WGd_0))
    X_WCd_0 = X_WGd_0.multiply(X_GC)
    U_candidates = []
    for displacement in _compute_displacements(density):
        X_WCd_i = X_WCd_0.multiply(
            RigidTransform(mr.MatrixExp6(mr.VecTose3(displacement)))
        )
        u_i = components.CompliantMotion(X_GC, X_WCd_i, K)
        U_candidates.append(u_i)

    P_results = dynamics.f_cspace(p, U_candidates)
    for idx, p_r in enumerate(P_results):
        if p_r.satisfies_contact(CF_d):
            if vis:
                dynamics.simulate(p, U_candidates[idx], vis=True)
            # print("good displacement: ", _compute_displacements(density)[idx])
            U.append(U_candidates[idx])
    return U


def _project_down(vertices: List[List[np.ndarray]]):
    """Generates polyhedra with positive measure from collections of vectors in R7.

    This algorithm starts with 7 dimensional vectors and tries to project them into lower
    dimensions until a dimensionality is found where each convex hull (generated
    from vertices[i]) has positive volume.

    Args:
        vertices: A list of lists of r7 vectors. Each "inner" list of vectors corresponds to
        edge vertices of a single polyhedron.

    Returns:
        A PCA object which can be used to map from the low-dim subspace back to R7, and
        the convex hulls in the low dimensional subspace.
    """
    n_verts = [len(poly) for poly in vertices]
    vertices_all = []
    for poly in vertices:
        vertices_all.extend(poly)
    ub = min(6, len(vertices_all))
    vertices_all = np.array(vertices_all)
    for n_components in range(ub, 1, -1):
        pca = PCA(n_components=n_components)
        low_dim_vertices = pca.fit_transform(vertices_all)
        curr_idx = 0
        polys = []
        successfully_projected = True
        for poly_idx in range(len(vertices)):
            poly_corners = low_dim_vertices[curr_idx : curr_idx + n_verts[poly_idx]]
            curr_idx += n_verts[poly_idx]
            try:
                polys.append(HPolyhedron(VPolytope(poly_corners.T)))
                vol = polys[-1].MaximumVolumeInscribedEllipsoid().Volume()
                if vol < 1e-4 and n_components > 2:
                    successfully_projected = False
            except:
                successfully_projected = False
        if successfully_projected:
            break
    print(f"dimensionality of low dimensional manifold = {n_components}")
    return pca, polys


def _sample_from_polyhedron(poly: HPolyhedron, n_samples: int = 8) -> List[np.ndarray]:
    samples = []
    rng = RandomGenerator()
    sample_i = poly.UniformSample(rng)
    for i in range(400):  # allow hit and run sampling time to converge/mix?
        sample_i = poly.UniformSample(rng, sample_i)
    for i in range(n_samples):
        samples.append(poly.UniformSample(rng, sample_i))
        sample_i = samples[-1]
    return samples


def naive_center(hulls: List[HPolyhedron]) -> np.ndarray:
    scale = 1.0 / len(hulls)
    components = [
        scale * hull.MaximumVolumeInscribedEllipsoid().center() for hull in hulls
    ]
    return sum(components)


def naive_center2(hulls: List[HPolyhedron]) -> np.ndarray:
    scale = 1.0 / len(hulls)
    components = [scale * hull.ChebyshevCenter() for hull in hulls]
    return sum(components)


def naive_center3(hulls: List[HPolyhedron], i: int = 1) -> np.ndarray:
    assert i < len(hulls)
    try:
        v = VPolytope(hulls[i])
        vtx = v.vertices()[:, 0]
        return vtx
    except Exception as e:
        print(f"nc3 failed with {e}")
        return naive_center(hulls)


def intersect_motion_sets(
    X_GC: RigidTransform,
    K: np.ndarray,
    b: state.Belief,
    CF_d: components.ContactState,
) -> List[components.CompliantMotion]:

    # grow motion set for each particle
    motion_sets = [grow_motion_set(X_GC, K, CF_d, p, vis=False) for p in b.particles]
    u_nom = motion_sets[0][0]
    motion_sets_unpacked = [cm for mset in motion_sets for cm in mset]
    # extract the setpoint from each CompliantMotion object in each motion set
    target_sets = [[u.X_WCd for u in motion_set] for motion_set in motion_sets]
    # convert setpoints from 4x4 matrix repr to 6-dimensional (se(3), xyz) vectors
    vertices = logmap_setpoints(target_sets)
    for vset in vertices:
        print(f"{len(vset)=}")
        if len(vset) < 2:  # can't have a positive-volume polytope from 0 or 1 points
            print("merge failed, 0 measure set detected")
            return random.sample(
                motion_sets_unpacked, min(8, len(motion_sets_unpacked))
            )
    # project vertex set to shared subspace where their convex hulls have positive measure
    mapping, hulls = _project_down(vertices)
    # visualize.plot_motion_sets(hulls)
    # intersect hulls
    intersection = hulls[0].Intersection(hulls[1])
    for i in range(2, len(hulls)):
        intersection = intersection.Intersection(hulls[i])
    if intersection.IsEmpty():
        print("merge failed, no intersection found")
        u_c1 = sample_to_motion(naive_center(hulls), target_sets[0][0], mapping, u_nom)
        u_c2 = sample_to_motion(naive_center2(hulls), target_sets[0][0], mapping, u_nom)
        u_c3 = sample_to_motion(naive_center3(hulls), target_sets[0][0], mapping, u_nom)
        u_c4 = sample_to_motion(
            naive_center3(hulls, i=0), target_sets[0][0], mapping, u_nom
        )
        print("u_c1 = ", utils.rt_to_str(u_c1.X_WCd))
        candidates = [u_c1, u_c2, u_c3, u_c4]
        candidates_clean = [c for c in candidates if (not c.has_nan())]
        return random.sample(motion_sets_unpacked, 8)
    # draw points from the hull intersection, use it to populate CompliantMotion objects
    naive_motion = sample_to_motion(
        naive_center(hulls), target_sets[0][0], mapping, u_nom
    )
    naive_motion2 = sample_to_motion(
        naive_center2(hulls), target_sets[0][0], mapping, u_nom
    )
    nm3_0 = sample_to_motion(
        naive_center3(hulls, i=0), target_sets[0][0], mapping, u_nom
    )
    nm3_1 = sample_to_motion(naive_center3(hulls), target_sets[0][0], mapping, u_nom)
    X_WCd_center_low_dim = intersection.MaximumVolumeInscribedEllipsoid().center()
    center_motion = sample_to_motion(
        X_WCd_center_low_dim, target_sets[0][0], mapping, u_nom
    )
    samples = _sample_from_polyhedron(intersection, n_samples=3)
    sampled_motions = [
        sample_to_motion(sample, target_sets[0][0], mapping, u_nom)
        for sample in samples
    ]
    candidates = [nm3_1, nm3_0, center_motion] + sampled_motions
    candidates_clean = [c for c in candidates if (not c.has_nan())]
    for c in candidates_clean:
        print(f"{c.X_WCd=}")
    return candidates_clean


def principled_intersect(
    X_GC: RigidTransform,
    K: np.ndarray,
    b: state.Belief,
    CF_d: components.ContactState,
) -> List[components.CompliantMotion]:
    # grow motion set for each particle
    motion_sets = [grow_motion_set(X_GC, K, CF_d, p) for p in b.particles]
    motion_sets_unpacked = [cm for mset in motion_sets for cm in mset]
    # extract the setpoint from each CompliantMotion object in each motion set
    target_sets = [[u.X_WCd for u in motion_set] for motion_set in motion_sets]
    breakpoint()
    vertices = logmap_setpoints(target_sets)
    hulls = []
    for poly in vertices:
        poly_mat = np.array(poly)
        v = VPolytope(poly_mat.T)
        hulls.append(HPolyhedron(v))
    intersection = hulls[0].Intersection(hulls[1])
    for i in range(2, len(hulls)):
        intersection = intersection.Intersection(hulls[i])
    if intersection.IsEmpty():
        raise Exception("merge failed: no intersection found")
    X_WCd_log = intersection.MaximumVolumeInscribedEllipsoid().center()
    R_bar = mr.MatrixExp3(mr.VecToso3(X_WCd_log[:3]))
    R = RotationMatrix(R_bar).multiply(target_sets[0][0].rotation())
    X_WCd = RigidTransform(R, X_WCd_log[3:])
    u_nom = motion_sets[0][0]
    center_motion = [components.CompliantMotion(u_nom.X_GC, X_WCd, u_nom.K)]
    return center_motion
