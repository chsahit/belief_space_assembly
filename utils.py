import pickle
from typing import List, Tuple

import numpy as np
import trimesh
from pydrake.all import Quaternion, RigidTransform, RollPitchYaw, RotationMatrix

import components


def xyz_rpy_deg(xyz: List[float], rpy_deg: List[float]) -> RigidTransform:
    """Shorthand for defining a pose."""
    rpy_deg_np = np.asarray(rpy_deg)
    xyz_np = np.asarray(xyz)
    return RigidTransform(RollPitchYaw(rpy_deg_np * np.pi / 180), xyz_np)


def RigidTfToVec(X: RigidTransform) -> np.ndarray:
    quat = X.rotation().ToQuaternion()
    quat = np.array([quat.w(), quat.x(), quat.y(), quat.z()])
    return np.concatenate((quat, X.translation()))


def VecToRigidTF(v: np.ndarray) -> RigidTransform:
    quat = Quaternion(v[:4] / np.linalg.norm(v[:4]))
    return RigidTransform(RotationMatrix(quat), v[4:])


def rt_to_str(X: RigidTransform) -> str:
    t_str = f"translation: {np.round(X.translation(), 5)}"
    r_str = f"rotation: {np.round(X.rotation().ToRollPitchYaw().vector(), 5)}"
    return t_str + "\n" + r_str


def median_mad(vals: np.ndarray) -> Tuple[float, float]:
    median = np.median(vals)
    mad = np.median(np.absolute(vals - median))
    return median, mad


def result_statistics(results):
    np_list = np.array([result.num_posteriors for result in results])
    mu_np, mu_std = median_mad(np_list)
    wall_time_list = np.array([result.total_time for result in results])
    mu_walltime, std_walltime = median_mad(wall_time_list)
    sim_time_list = np.array([result.sim_time for result in results])
    mu_sim_time, std_sim_time = median_mad(sim_time_list)

    succs = [r.traj for r in results if r.traj is not None]
    sr = float(len(succs)) / len(results)
    return (mu_np, mu_std), (mu_walltime, std_walltime), (mu_sim_time, std_sim_time), sr


def dump_mesh(mesh: trimesh.Trimesh, fname: str = "logs/cspace.obj"):
    joined_mesh_obj = mesh.export(file_type="obj")
    with open(fname, "w") as f:
        f.write(joined_mesh_obj)


def log_experiment_result(
    fname: str,
    experiment_label: str,
    experiment_results: List[components.PlanningResult],
):
    try:
        with open(fname, "rb") as f:
            results = pickle.load(f)
    except Exception:
        results = dict()
    results[experiment_label] = experiment_results
    with open(fname, "wb") as f:
        pickle.dump(results, f)
    del results


def pickle_trajectory(
    p0,
    traj: List[components.CompliantMotion],
    fname: str = "logs/traj_out.pkl",
    joints: bool = True,
):
    data = []
    u0 = components.CompliantMotion(
        RigidTransform(), p0.X_WG, np.diag(components.very_stiff)
    )
    traj_ = [u0] + traj
    for u in traj_:
        K = np.copy(u.K)
        K_r = np.copy(K[:3, :3])
        K[:3, :3] = K[3:, 3:]
        K[3:, 3:] = K_r
        K_flat = K.flatten().tolist()
        if joints:
            command = u.q_d[:7].tolist()
            data.append((command, K_flat))
        else:
            command_vec = RigidTfToVec(u.X_WCd)
            quat = [command_vec[0], command_vec[1], command_vec[2], command_vec[3]]
            data.append((quat, command_vec[4:], K_flat))

    with open(fname, "wb") as f:
        pickle.dump(data, f)
