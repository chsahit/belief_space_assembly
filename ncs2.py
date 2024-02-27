from collections import dataclass

from pydrake.all import HPolyhedron, RigidTransform, VPolytope

import components


@dataclass
class WorkspaceObject:
    name: str
    geometry: List[HPolyhedron]


class CFace:
    def __init__(self, label: components.ContactState, eqn: HPolyhedron):
        self.label = label
        self.eqn = eqn

    def PointOnFace(self, pt: np.ndarray) -> bool:
        residual = np.abs(self.eqn.A() @ pt - self.eqn.b())
        return (residual < 1e-6).all()


class CObs:
    def __init__(self, faces: List[CFace]):
        self.faces = faces

    def get_contact(self, pt: np.ndarray) -> components.ContactState:
        label = ()


def sample_pose(contact: components.ContactState, cspace: List[CObs]) -> np.ndarray:
    pass


def make_cspace(
    manip: List[WorkspaceObject],
    env: List[WorkspaceObject],
    X_WM: RigidTransform,
    X_WO: RigidTransform,
) -> List[CObs]:
    pass


def compute_task_plan(
    cspace: List[CObs], goal: components.ContactState
) -> List[components.ContactState]:
    pass
