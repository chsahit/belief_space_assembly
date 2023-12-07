from typing import List

from pydrake.all import HPolyhedron, RigidTransform

import components
import state
from simulation import annotate_geoms


def compute_contact_set(
    p: state.Particle, CF_d: components.ContactState
) -> HPolyhedron:
    constraints = p.constraints
    corner_map = annotate_geoms.annotate(p.manip_geom)
    hp = None
    for env_poly, object_corner in CF_d:
        X_MP = corner_map[object_corner]
        A, b = constraints[env_poly]
        A_translated = A @ X_MP.GetAsMatrix4()
        if hp is None:
            hp = HPolyhedron(A_translated, b)
        else:
            curr_hp = HPolyhedron(A_translated, b)
            hp = hp.Intersection(curr_hp)
    return hp


def compute_samples_from_contact_set(
    p: state.Particle, CF_d: components.ContactState
) -> List[RigidTransform]:
    pass
