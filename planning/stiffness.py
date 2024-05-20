import numpy as np
import trimesh
from pydrake.all import RigidTransform, RotationMatrix
from trimesh import proximity

import components
import cspace
import mr
import state

env = None


def normal_vec_to_matrix(n: np.ndarray) -> np.ndarray:
    vecs = np.concatenate((np.array([n]), np.eye(3)))
    q, r = np.linalg.qr(vecs.T)
    q[:, 0] = n
    basis_vectors = q[:, [1, 2, 0]]
    return basis_vectors


def approximate_cspace_gradient(axis: int, p: state.Particle) -> np.ndarray:
    current_translation = np.array([p.X_WM.translation()])
    drotation = np.array([0.0, 0.0, 0.0])
    drotation[axis] = 0.02
    drotation_SE3 = RotationMatrix(mr.MatrixExp3(mr.VecToso3(drotation)))
    R_WM = p.X_GM.rotation().multiply(drotation_SE3)
    cspace_surface = cspace.ConstructCspaceSlice(env, R_WM).mesh
    closest_pt, _, _ = proximity.closest_point(cspace_surface, current_translation)
    d_drotation = np.linalg.norm(closest_pt - current_translation) / drotation[axis]
    return d_drotation


def translational_normal(X_WM: RigidTransform, mesh: trimesh.Trimesh) -> np.ndarray:
    current_translation = np.array([X_WM.translation()])
    _, dist, triangle_id = proximity.closest_point(mesh, current_translation)
    if dist[0] > 0.01:
        return None
    translational_normal = mesh.face_normals[triangle_id][0]
    return translational_normal


def solve_for_compliance(p: state.Particle) -> np.ndarray:
    global env
    if env is None:
        env = cspace.ConstructEnv(p)
    p.cspace_repr = cspace.ConstructCspaceSlice(env, p.X_WM.rotation()).mesh
    cspace_surface = p.cspace_repr
    translational_normal = make_translational_normal(p, cspace_surface)
    if translational_normal is None:
        return ablate_compliance()

    R_CW = normal_vec_to_matrix(translational_normal)
    # K_t_diag = np.copy(components.stiff[3:])
    # K_t_diag[2] = components.soft[5]
    K_t_diag = np.copy(components.very_stiff[3:])
    K_t_diag[2] = components.stiff[5]
    K_t = R_CW @ np.diag(K_t_diag) @ np.linalg.inv(R_CW)
    rotational_normal = np.array([approximate_cspace_gradient(i, p) for i in range(3)])
    if np.linalg.norm(rotational_normal) > 1e-8:
        rotational_normal = rotational_normal / np.linalg.norm(rotational_normal)
    R_CW_rot = normal_vec_to_matrix(rotational_normal)
    # K_r_diag = np.copy(components.stiff[:3])
    # K_r_diag[2] = components.soft[2]
    K_r_diag = np.copy(components.very_stiff[:3])
    K_r_diag[2] = components.stiff[2]
    K_r = R_CW_rot @ np.diag(K_r_diag) @ np.linalg.inv(R_CW_rot)
    if np.linalg.norm(rotational_normal) < 1e-8:
        K_r = np.diag(np.array([60.0, 60.0, 60.0]))
    K = np.zeros((6, 6))
    K[:3, :3] = K_r  # np.diag(components.stiff[:3])
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
    K[:3, :3] = np.diag(components.very_stiff[:3])  # K_r
    K[3:, 3:] = np.diag(components.very_stiff[3:])  # K_t
    return K, []
