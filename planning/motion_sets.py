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
        if i < 3:
            Delta = np.linspace(-0.25, 0.25, density).tolist()
        else:
            Delta = np.linspace(-0.03, 0.03, density).tolist()
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
            R_WC = nominal
            delta = nominal.multiply(R_WC.inverse()).matrix()
            r = mr.so3ToVec(mr.MatrixLog3(delta))
            logmapped.append(np.concatenate((r, t)))
        logmap_batch.append(logmapped)
    return logmap_batch


def expmap_intersection(sp: np.ndarray, origin: RigidTransform) -> RigidTransform:
    R_bar = mr.MatrixExp3(mr.VecToso3(sp[:3]))
    R = RotationMatrix(R_bar).multiply(origin.rotation())
    return RigidTransform(R, sp[3:])


def grow_motion_set(
    X_GC: RigidTransform,
    K: np.ndarray,
    CF_d: components.ContactState,
    p: state.Particle,
    density: int = 5,
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
    for idx, p in enumerate(P_results):
        if p.satisfies_contact(CF_d):
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
    vertices_all = np.array(vertices_all)
    for n_components in range(6, 1, -1):
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


def _sample_from_polyhedron(poly: HPolyhedron, n_samples: int = 10) -> List[np.ndarray]:
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


def intersect_motion_sets(
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
    # convert setpoints from 4x4 matrix repr to 7-dimensional (quat, xyz) vectors
    """
    vertices = [
        [utils.RigidTfToVec(X_WCd) for X_WCd in target_set]
        for target_set in target_sets
    ]
    """
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
        # c0 = hulls[0].MaximumVolumeInscribedEllipsoid().center()
        # c1 = hulls[1].MaximumVolumeInscribedEllipsoid().center()
        # c = 0.5 * c0 + 0.5 * c1
        c = naive_center(hulls)
        X_WCd = expmap_intersection(
            mapping.inverse_transform([c][0]), target_sets[0][0]
        )
        u_nom = motion_sets[0][0]
        u = components.CompliantMotion(u_nom.X_GC, X_WCd, u_nom.K)
        return [u]
        return random.sample(motion_sets_unpacked, 8)
    # draw points from the hull intersection, use it to populate CompliantMotion objects
    u_nom = motion_sets[0][0]
    X_WCd_center_low_dim = intersection.MaximumVolumeInscribedEllipsoid().center()
    """
    X_WCd_center = utils.VecToRigidTF(
        mapping.inverse_transform([X_WCd_center_low_dim][0])
    )
    """
    X_WCd_naive = expmap_intersection(
        mapping.inverse_transform([naive_center(hulls)][0]), target_sets[0][0]
    )
    naive_motion = [components.CompliantMotion(u_nom.X_GC, X_WCd_naive, u_nom.K)]
    X_WCd_center = expmap_intersection(
        mapping.inverse_transform([X_WCd_center_low_dim][0]), target_sets[0][0]
    )
    center_motion = [components.CompliantMotion(u_nom.X_GC, X_WCd_center, u_nom.K)]
    samples = _sample_from_polyhedron(intersection)
    """
    samples_rt = [
        utils.VecToRigidTF(mapping.inverse_transform([sample])[0]) for sample in samples
    ]"""
    samples_rt = [
        expmap_intersection(mapping.inverse_transform([sample][0]), target_sets[0][0])
        for sample in samples
    ]
    sampled_motions = [
        components.CompliantMotion(u_nom.X_GC, sample_rt, u_nom.K)
        for sample_rt in samples_rt
    ]
    return naive_motion + center_motion + sampled_motions


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
