from dataclasses import dataclass
from typing import FrozenSet, List, NamedTuple, Tuple

import numpy as np
from pydrake.all import RigidTransform, RollPitchYaw

Contact = Tuple[str, str]
ContactState = FrozenSet[Contact]

stiff = np.array([60.0, 60.0, 60.0, 600.0, 600.0, 600.0])
soft = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 100.0])


@dataclass
class CompliantMotion:
    X_GC: RigidTransform
    X_WCd: RigidTransform
    K: np.ndarray
    _B: np.ndarray = None
    timeout: float = 5.0
    is_joint_space: bool = False
    q_d: np.ndarray = None

    @property
    def B(self) -> np.ndarray:
        if self._B is not None:
            return self._B
        else:
            return 4 * np.sqrt(self.K)

    def has_nan(self) -> bool:
        r1 = np.isnan(self.X_GC.rotation().matrix()).any()
        r2 = np.isnan(self.X_WCd.rotation().matrix()).any()
        t1 = np.isnan(self.X_GC.translation()).any()
        t2 = np.isnan(self.X_WCd.translation()).any()
        k = np.isnan(self.K).any()
        return r1 or r2 or t1 or t2 or k


class Grasp(NamedTuple):
    x: float
    z: float
    pitch: float
    roll: float = 0

    def get_tf(self):
        return RigidTransform(
            RollPitchYaw(
                np.array([self.roll * np.pi / 180, self.pitch * np.pi / 180, 0])
            ),
            [self.x, 0, self.z],
        )


class ObjectPose(NamedTuple):
    x: float
    y: float
    yaw: float

    def get_tf(self):
        return RigidTransform(
            RollPitchYaw(np.array([0, 0, self.yaw * np.pi / 180])),
            [self.x, self.y, 0.075],
        )


@dataclass(frozen=True)
class TreeNode:
    u: CompliantMotion
    u_pred: "TreeNode"
    score: int

    @property
    def r7_repr(self) -> np.ndarray:
        # this is copy pasted from utils.py :(
        X = self.u.X_WCd
        quat = X.rotation().ToQuaternion()
        quat = np.array([quat.w(), quat.x(), quat.y(), quat.z()])
        return np.concatenate((quat, X.translation()))


@dataclass
class Tree:
    p: "Particle"
    nodes: List[TreeNode]


@dataclass
class PlanningResult:
    traj: List[CompliantMotion]
    total_time: float
    sim_time: float
    num_posteriors: int

    def __str__(self):
        if self.traj is None:
            traj_len = 0
        else:
            traj_len = len(self.traj)
        return f"{self.total_time=}, {self.num_posteriors=}, {traj_len=}"
