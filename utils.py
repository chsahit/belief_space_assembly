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


def dump_traj(traj: List[components.CompliantMotion], fname: str = "traj_out.pkl"):
    tau = []
    for u in traj:
        X_WGd = u.X_WCd.multiply(u.X_GC.inverse())
        u_pkl = (X_WGd.GetAsMatrix4(), u.K)
        tau.append(u_pkl)
    with open(fname, "wb") as f:
        pickle.dump(tau, f)


def post_process_rt_pickle(fname: str = "traj_out.pkl"):
    processed = []
    with open(fname, "rb") as f:
        data = pickle.load(f)
        for command in data:
            K = command[1]
            K = [K[3], K[4], K[5], K[0], K[1], K[2]]
            command_vec = RigidTfToVec(RigidTransform(command[0]))
            # w, x y, z -> x, y, z, w
            quat = [command_vec[1], command_vec[2], command_vec[3], command_vec[0]]
            quat = [command_vec[0], command_vec[1], command_vec[2], command_vec[3]]
            processed.append((quat, command_vec[4:], K))
    with open(fname, "wb") as f:
        pickle.dump(processed, f)


if __name__ == "__main__":
    post_process_rt_pickle("rot_uncertain.pkl")
