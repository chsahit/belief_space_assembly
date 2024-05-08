import pickle
from typing import List

import numpy as np
import trimesh
from pydrake.all import (
    HPolyhedron,
    Quaternion,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    VPolytope,
)

import components


def xyz_rpy_deg(xyz: List[float], rpy_deg: List[float]) -> RigidTransform:
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)


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


def dump_traj(
    init_q: np.ndarray,
    traj: List[components.CompliantMotion],
    fname: str = "traj_out.pkl",
):
    tau = [(init_q, components.stiff)]
    for u in traj:
        X_WGd = u.X_WCd.multiply(u.X_GC.inverse())
        u_pkl = (X_WGd.GetAsMatrix4(), u.K)
        tau.append(u_pkl)
    with open(fname, "wb") as f:
        pickle.dump(tau, f)


def mu_std_result(results):
    traj_lens = []
    succs = 0.0
    for r in results:
        if r.traj is not None:
            succs += 1.0
            traj_lens.append(len(r.traj))
        """
        else:
            traj_lens.append(20)
        """
    if len(traj_lens) == 0:
        print("setting to timeout")
        traj_lens = [50]
    times = np.array(traj_lens)
    times = np.array([result.num_posteriors for result in results])
    mu, std = np.mean(times), np.std(times)
    sr = succs / len(results)
    return mu, std, sr


def envelope_analysis(data):
    ours_max = float("-inf")
    no_stiffness_max = float("-inf")
    no_gp_max = float("-inf")
    for params, results in data.items():
        has_success = any([result.traj is not None for result in results])
        if params[1] == "True" and params[2] == "True" and has_success:
            ours_max = max(ours_max, 2 * float(params[0]))
        elif params[1] == "True" and params[2] == "False" and has_success:
            no_stiffness_max = max(no_stiffness_max, 2 * float(params[0]))
        elif params[1] == "False" and params[2] == "True" and has_success:
            no_gp_max = max(no_gp_max, 2 * float(params[0]))

    print(f"{ours_max=}")
    print(f"{no_stiffness_max=}")
    print(f"{no_gp_max=}")


def GetVertices(H: HPolyhedron, assert_count: bool = True) -> np.ndarray:
    try:
        V = VPolytope(H.ReduceInequalities(tol=1e-6))
    except SystemExit:
        try:
            V = VPolytope(H.ReduceInequalities(tol=1e-4))
        except SystemExit as se:
            print(f"SystemExit {se} from getvertices")
            breakpoint()
            return None
    vertices = V.vertices().T
    return vertices


def label_to_str(label: components.ContactState) -> str:
    tag = ""
    for contact in label:
        A = contact[0][contact[0].find("::") + 2 :]
        B = contact[1][contact[1].find("::") + 2 :]
        tag += f"({A}, {B}), "
    return tag


def dump_mesh(mesh: trimesh.Trimesh):
    joined_mesh_obj = mesh.export(file_type="obj")
    with open("cspace.obj", "w") as f:
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
    traj: List[components.CompliantMotion],
    fname: str = "traj_out.pkl",
    joints: bool = True,
):
    data = []
    for u in traj:
        K = np.diag(u.K).tolist()
        if joints:
            command = u.q_d[:7].tolist()
            data.append((command, K))
        else:
            command_vec = RigidTfToVec(u.X_WCd)
            quat = [command_vec[0], command_vec[1], command_vec[2], command_vec[3]]
            data.append((quat, command_vec[4:], K))

    with open(fname, "wb") as f:
        pickle.dump(data, f)
