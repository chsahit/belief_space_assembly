import pickle
from typing import List

import numpy as np
from pydrake.all import Quaternion, RigidTransform, RollPitchYaw, RotationMatrix

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
    times = np.array([result.total_time for result in results])
    mu, std = np.mean(times), np.std(times)
    return mu, std


if __name__ == "__main__":
    post_process_rt_pickle("rot_uncertain.pkl")
