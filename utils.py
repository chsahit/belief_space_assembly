from typing import List

import numpy as np
from pydrake.all import RigidTransform, RollPitchYaw


def xyz_rpy_deg(xyz: List[float], rpy_deg: List[float]) -> RigidTransform:
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)
