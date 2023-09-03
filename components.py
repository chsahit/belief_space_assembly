from dataclasses import dataclass
from typing import FrozenSet, Tuple

import numpy as np
from pydrake.all import RigidTransform

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
            return 4 * np.sqrt(self.K)
