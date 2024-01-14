import pickle

from pydrake.all import RigidTransform

import utils
from simulation import ik_solver


def post_process_rt_pickle(fname: str = "traj_out.pkl"):
    processed = []
    with open(fname, "rb") as f:
        data = pickle.load(f)
        for command in data:
            K = command[1]
            K = [K[3], K[4], K[5], K[0], K[1], K[2]]
            command_vec = utils.RigidTfToVec(RigidTransform(command[0]))
            quat = [command_vec[0], command_vec[1], command_vec[2], command_vec[3]]
            processed.append((quat, command_vec[4:], K))
    with open(fname, "wb") as f:
        pickle.dump(processed, f)


def joint_space_post_process(fname: str):
    processed = []
    with open(fname, "rb") as f:
        data = pickle.load(f)
        processed.append((data[0][0], [400, 400, 400, 60, 60, 60]))
        for X_WGd, K in data[1:]:
            K_q = [K[3], K[4], K[5], K[0], K[1], K[2]]
            q_r = ik_solver.gripper_to_joint_states(RigidTransform(X_WGd))
            processed.append((q_r, K_q))
    with open("joint_" + fname, "wb") as f_write:
        pickle.dump(processed, f_write)


if __name__ == "__main__":
    joint_space_post_process("rot_uncertain.pkl")
