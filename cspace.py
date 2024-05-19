import itertools
from collections import defaultdict
from typing import List

import cdd
import networkx as nx
import numpy as np
import trimesh
from pydrake.all import HPolyhedron, RotationMatrix, VPolytope
from scipy.spatial import ConvexHull

import components
import contact_defs
import state

cross_section_cache = []


def is_face(name: str) -> bool:
    suffixes = ["_bottom", "_top", "_left", "_right", "_front", "_back", "_inside"]
    is_face = any([suffix in name for suffix in suffixes])
    return is_face


def ConstructEnv(p: state.Particle) -> components.Env:
    Obj = dict()
    for name, poly in p.constraints.items():
        if not is_face(name):
            Obj[name] = poly
    M = dict()
    for name, poly in p._manip_poly.items():
        if not is_face(name):
            M[name] = poly
    return components.Env(M, Obj)


def tf_hrepr(H: components.HRepr, R: RotationMatrix) -> HPolyhedron:
    A_tf = H[0] @ R.inverse().matrix()
    return HPolyhedron(A_tf, H[1])


def MatToArr(m: cdd.Matrix) -> np.ndarray:
    return np.array([m[i] for i in range(m.row_size)])


def toGenerators(H: HPolyhedron) -> np.ndarray:
    A, b = (H.A(), H.b())
    H_repr = np.hstack((np.array([b]).T, -A))
    mat = cdd.Matrix(H_repr, number_type="float")
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    V_repr = MatToArr(poly.get_generators())
    vertices = V_repr[:, 1:]
    return vertices


def minkowski_difference(B: HPolyhedron, A: HPolyhedron) -> VPolytope:
    B_vertices = toGenerators(B)
    A_vertices = toGenerators(A)
    volume_verts = []
    for vB_idx in range(B_vertices.shape[0]):
        for vA_idx in range(A_vertices.shape[0]):
            volume_verts.append(B_vertices[vB_idx] - A_vertices[vA_idx])
    return VPolytope(np.array(volume_verts).T)


def ConstructCspaceSlice(
    env: components.Env, rotation: RotationMatrix, use_cache: bool = True
) -> components.CSpaceSlice:
    global cross_section_cache
    if use_cache:
        for csec in cross_section_cache:
            diff = rotation.InvertAndCompose(csec.rot)
            if diff.IsNearlyIdentity():
                return csec
    # rotate manipuland
    M_tf = dict()
    for name, poly in env.M.items():
        M_tf[name] = tf_hrepr(poly, rotation)
    # compute minkowski differences
    cspace_vols = []
    for O_name, O_poly in env.O.items():
        O_H = HPolyhedron(*O_poly)
        for M_name, M_poly in M_tf.items():
            cspace_vols.append(minkowski_difference(O_H, M_poly))
    # convert into faces and simplices for trimesh
    cspace_vol_verts = [vol.vertices().T for vol in cspace_vols]
    hulls = [(v, ConvexHull(v).simplices) for v in cspace_vol_verts]
    # make trimesh repr
    meshes = [trimesh.Trimesh(vertices=hull[0], faces=hull[1]) for hull in hulls]
    for mesh in meshes:
        mesh.fix_normals()
        mesh.process()
    joined_mesh = meshes[0]
    for i, mesh in enumerate(meshes[1:]):
        joined_mesh = joined_mesh.union(mesh)

    cspace_slice = components.CSpaceSlice(joined_mesh, rotation)
    if use_cache and len(cross_section_cache) < 10:
        cross_section_cache.append(cspace_slice)
    return cspace_slice


def label_mesh(
    cspace: components.CSpaceSlice, p: state.Particle
) -> components.TaskGraph:
    # make dense volume collection (expensive!)
    Obj_faces = dict()
    M_faces = dict()
    for name, poly in p.constraints.items():
        if is_face(name):
            Obj_faces[name] = HPolyhedron(*poly)
    for name, poly in p._manip_poly.items():
        if is_face(name):
            M_faces[name] = tf_hrepr(poly, p.X_WM.rotation())
    labeled_volumes = []
    for O_name, O_poly in Obj_faces.items():
        for M_name, M_poly in M_faces.items():
            label = frozenset(((O_name, M_name),))
            volume = minkowski_difference(O_poly, M_poly)
            labeled_volumes.append((label, volume))
    # for each vertex on mesh, check which contacts they satisfy. also build normal map
    vertex_labels = defaultdict(list)
    normal_map = defaultdict(list)
    label_to_verts = defaultdict(list)
    for i, vertex in enumerate(cspace.mesh.vertices):
        for label, volume in labeled_volumes:
            if volume.PointInSet(vertex):
                vertex_labels[i].append(label)
                normal_map[label].append(cspace.mesh.vertex_normals[i])
                label_to_verts[label].append(vertex)
    # prune contact modes that are degenerate
    V = []
    for labeled_volume in labeled_volumes:
        if len(label_to_verts[labeled_volume[0]]) >= 2:
            V.append(labeled_volume[0])
    # for each vertex, connect each contact associated with that vertex
    edges = set()
    for vtx, labels in vertex_labels.items():
        for edge in itertools.combinations(labels, 2):
            if str(edge[0]) < str(edge[1]):
                edge = (edge[1], edge[0])
            if edge[0] in V and edge[1] in V:
                edges.add(edge)
    G = components.TaskGraph(V, edges, normal_map, label_to_verts)
    return G


def make_cost_fn(uncertainty_dir: np.ndarray, G: components.TaskGraph):
    normal_map = G.g_normal

    def cost(v1: components.ContactState, v2: components.ContactState, attrs) -> float:
        if v2 not in normal_map.keys():
            return 1
        projection = np.dot(normal_map[v2][0], uncertainty_dir) * uncertainty_dir
        bonus = 1e-4 * np.linalg.norm(projection)
        return 1 - bonus

    return cost


def make_task_plan(
    G: components.TaskGraph,
    start_mode: components.ContactState,
    goal_mode: components.ContactState,
    uncertainty_dir: np.ndarray,
    start_pose: np.ndarray = None,
) -> List[components.ContactState]:
    # decide on free space neighbors
    n_closest = 2
    if start_mode == contact_defs.fs:
        assert start_pose is not None
        distances = []
        for v in G.V:
            distances.append(np.linalg.norm(start_pose - G.repr_points[v]))
        smallest_indices = np.argpartition(distances, n_closest)
        fs_neighbors = []
        for i in range(n_closest):
            fs_neighbors.append((contact_defs.fs, G.V[smallest_indices[i]]))
        E_fs = list(G.E) + fs_neighbors
    else:
        E_fs = G.E
    # convert taskgraph to networkx graph
    nx_graph = nx.Graph()
    for e in E_fs:
        nx_graph.add_edge(*e)
    nx_graph = nx_graph.to_directed()
    # search on graph, order paths lexicographically so experiments are deterministic
    cost_fn = make_cost_fn(uncertainty_dir, G)
    try:
        candidate_paths = list(
            nx.all_shortest_paths(
                nx_graph, source=start_mode, target=goal_mode, weight=cost_fn
            )
        )
    except Exception as e:
        print(f"search exception={e}")
        return None
    sorted_paths = sorted(list(candidate_paths), key=lambda sched: str(sched))
    contact_schedule = sorted_paths[0]
    return contact_schedule
