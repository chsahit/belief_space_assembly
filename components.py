from dataclasses import dataclass
from typing import FrozenSet, NamedTuple, Tuple

import numpy as np
from pydrake.all import RigidTransform, RollPitchYaw

Contact = Tuple[str, str]
ContactState = FrozenSet[Contact]

stiff = np.array([100.0, 100.0, 100.0, 600.0, 600.0, 600.0])
soft = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 100.0])


@dataclass
class CompliantMotion:
    X_GC: RigidTransform
    X_WCd: RigidTransform
    K: np.ndarray
    _B: np.ndarray = None
    timeout: float = 5.0

    @property
    def B(self):
        if self._B is not None:
            return self._B
        else:
            return 6 * np.sqrt(self.K)


class Grasp(NamedTuple):
    x: float
    z: float
    pitch: float

    def get_tf(self):
        return RigidTransform(
            RollPitchYaw(np.array([0, self.pitch * np.pi / 180, 0])),
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
