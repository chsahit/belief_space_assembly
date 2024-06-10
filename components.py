from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import trimesh
from pydrake.all import RigidTransform, RotationMatrix

from simulation import ik_solver

Contact = Tuple[str, str]
ContactState = FrozenSet[Contact]
Hull = Tuple[np.ndarray, List[List[float]]]
HRepr = Tuple[np.ndarray, np.ndarray]  # (A,b) s.t Ax <= b

very_stiff = np.array([100.0, 100.0, 100.0, 600.0, 600.0, 600.0])
stiff = np.array([60.0, 60.0, 60.0, 400.0, 400.0, 400.0])
soft = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 100.0])

very_stiff = np.array([60.0, 60.0, 60.0, 400.0, 400.0, 400.0])
stiff = np.array([30.0, 30.0, 30.0, 200.0, 200.0, 200.0])


@dataclass
class CompliantMotion:
    X_GC: RigidTransform
    X_WCd: RigidTransform
    K: np.ndarray
    _B: Optional[np.ndarray] = None
    timeout: float = 5.0
    is_joint_space: bool = False
    _q_d: Optional[np.ndarray] = None

    @property
    def B(self) -> np.ndarray:
        if self._B is not None:
            return self._B
        else:
            return 4 * np.sqrt(self.K)

    @property
    def q_d(self) -> np.ndarray:
        if self._q_d is None:
            X_WG = self.X_WCd.multiply(self.X_GC.inverse())
            self._q_d = ik_solver.gripper_to_joint_states(X_WG)
        return self._q_d


@dataclass
class PlanningResult:
    traj: Optional[List[CompliantMotion]]
    total_time: float
    sim_time: float
    num_posteriors: int
    last_refined: Optional[Tuple[ContactState, ContactState]]

    def __str__(self):
        if self.traj is None:
            traj_len = 0
        else:
            traj_len = len(self.traj)
        return f"{self.total_time=}, {self.num_posteriors=}, {traj_len=}"


@dataclass(frozen=True)
class Env:
    M: Dict[str, HRepr]
    O: Dict[str, HRepr]


@dataclass(frozen=True)
class CSpaceSlice:
    mesh: trimesh.Trimesh
    rot: RotationMatrix


@dataclass
class TaskGraph:
    V: List[ContactState]
    E: Set[Tuple[ContactState, ContactState]]
    g_normal: Dict[ContactState, List[np.ndarray]]
    repr_points: Dict[ContactState, List[np.ndarray]]
