from typing import List

from pydrake.all import HPolyhedron, RigidTransform

import components


def minkowski_sum(
    verts_B: List[components.WVert], verts_A: List[components.WVert]
) -> List[components.CVert]:
    verts = []
    for v_B in verts_B:
        for v_A in verts_A:
            verts.append(v_B - v_A)


def GeomVertices(A: HPolyhedron) -> List[np.ndarray]:
    pass


def label_geom(A: HPolyhedron, prefix: str) -> List[components.WVert]:
    pass


def make_full_cspace(
    manip: List[HPolyhedron],
    env: List[HPolyhedron],
    X_WM: RigidTransform,
    X_WO: RigidTransform,
):
    """Generates the CSpace obstacle for a given (obstacle, manipuland) pair.

    Given a manipuland and obstacle, both represented as a union of HPolyhedra(i.e., the semi-algebraic/half-space representation of a convex polyhedron), both of which have a known representation with respect to a global "world" frame - this compute the corresponding configuration space obstacle for a fixed orientation of the manipuland. Different faces, edges, and vertices of the obstacle correspond to differing contact constraints between the manipuland and obstacle.

    Args:
        manip: A list of pydrake HPolyhedron objects, the union of which defines the manipuland relative to the manipuland's body frame M.
        env: A list of pydrake HPolyhedron objects, the union of which defines an object in the environment relative to the object's body frame O.
        X_WM: The transformation from the world frame W to the manipuland body frame.
        X_WO: The transformation from the world frame W to the object body frame.


    Returns:
        A set of cspace faces as a graph (what type a graph is is tbd)
    """

    # represent the C-Obstacle as a union over the msum of each manip component with each env component
    C_obs_list = []
    for env_component in env:
        for manip_component in manip:
            pass
