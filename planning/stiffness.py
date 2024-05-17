import numpy as np
from pydrake.all import RotationMatrix
from trimesh import proximity

import components
import cspace
import mr
import state

rotation_to_trimesh_repr = dict()


def normal_vec_to_matrix(n: np.ndarray) -> np.ndarray:
    vecs = np.concatenate((np.array([n]), np.eye(3)))
    q, r = np.linalg.qr(vecs.T)
    q[:, 0] = n
    basis_vectors = q[:, [1, 2, 0]]
    return basis_vectors


def approximate_cspace_gradient(axis: int, p: state.Particle) -> np.ndarray:
    global rotation_to_trimesh_repr
    current_translation = np.array([p.X_WM.translation()])
    drotation = np.array([0.0, 0.0, 0.0])
    drotation[axis] = 0.05
    drotation_SE3 = RotationMatrix(mr.MatrixExp3(mr.VecToso3(drotation)))
    R_WM = p.X_GM.rotation().multiply(drotation_SE3)
    if axis not in rotation_to_trimesh_repr.keys():
        rotation_to_trimesh_repr[axis] = cspace.MakeTrimeshRepr(
            R_WM, p.constraints, p._manip_poly, do_tf=True
        )
    cspace_surface = rotation_to_trimesh_repr[axis]
    closest_pt, _, _ = proximity.closest_point(cspace_surface, current_translation)
    d_drotation = np.linalg.norm(closest_pt - current_translation) / drotation[axis]
    return d_drotation


def solve_for_compliance(p: state.Particle) -> np.ndarray:
    current_translation = np.array([p.X_WM.translation()])
    if p.cspace_repr is None:
        p.cspace_repr = cspace.MakeTrimeshRepr(
            p.X_WM.rotation(), p.constraints, p._manip_poly
        )
    cspace_surface = p.cspace_repr
    _, _, triangle_id = proximity.closest_point(cspace_surface, current_translation)
    translational_normal = cspace_surface.face_normals[triangle_id][0]
    R_CW = normal_vec_to_matrix(translational_normal)
    K_t_diag = np.copy(components.stiff[3:])
    K_t_diag[2] = components.soft[5]
    K_t = R_CW @ np.diag(K_t_diag) @ np.linalg.inv(R_CW)
    rotational_normal = np.array([approximate_cspace_gradient(i, p) for i in range(3)])
    if np.linalg.norm(rotational_normal) > 1e-8:
        rotational_normal = rotational_normal / np.linalg.norm(rotational_normal)
    R_CW_rot = normal_vec_to_matrix(rotational_normal)
    K_r_diag = np.copy(components.stiff[:3])
    K_r_diag[2] = components.soft[2]
    K_r = R_CW_rot @ np.diag(K_r_diag) @ np.linalg.inv(R_CW_rot)
    if np.linalg.norm(rotational_normal) < 1e-8:
        K_r = np.diag(np.array([60.0, 60.0, 60.0]))
    K = np.zeros((6, 6))
    K[:3, :3] = K_r
    K[3:, 3:] = K_t
    return K, []


def ablate_compliance() -> np.ndarray:
    normal = np.array([1, 1, 1])
    R_CW = normal_vec_to_matrix(normal)
    K_t_diag = np.copy(components.stiff[3:])
    K_t_diag[2] = components.soft[5]
    K_t = R_CW @ np.diag(K_t_diag) @ np.linalg.inv(R_CW)
    K_r_diag = np.copy(components.stiff[:3])
    K_r_diag[2] = components.soft[2]
    K_r = R_CW @ np.diag(K_r_diag) @ np.linalg.inv(R_CW)
    K = np.zeros((6, 6))
    K[:3, :3] = K_r
    K[3:, 3:] = K_t
    return K, []
