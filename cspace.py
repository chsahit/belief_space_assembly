import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import trimesh
from pydrake.all import HPolyhedron, RandomGenerator, RigidTransform, VPolytope
from scipy.spatial import ConvexHull

import components
import contact_defs
import graph
import puzzle_contact_defs
import utils

drake_rng = RandomGenerator(0)
gen = np.random.default_rng(1)


def TF_HPolyhedron(H: HPolyhedron, X_MMt: RigidTransform) -> HPolyhedron:
    vertices = utils.GetVertices(H)
    if vertices is None:
        breakpoint()
    tf = X_MMt.inverse().GetAsMatrix4()
    vertices_tf = []
    for v_idx in range(vertices.shape[0]):
        vert = vertices[v_idx]
        homogenous = np.array([vert[0], vert[1], vert[2], 1])
        homogenous_tf = tf @ homogenous
        vertices_tf.append(homogenous_tf[:3])
    return HPolyhedron(VPolytope(np.array(vertices_tf).T))


class CSpaceVolume:
    def __init__(self, label: components.ContactState, geometry: HPolyhedron):
        self.label = label
        self.geometry = geometry
        self._normal = None
        self._center = None

    def sample(self) -> np.ndarray:
        return self.geometry.UniformSample(drake_rng, mixing_steps=1000)

    @property
    def center(self):
        if self._center is None:
            verts = utils.GetVertices(self.geometry)
            if verts is None:
                return np.array([100, 100, 100])
            verts = verts.tolist()
            center_pt = sum(verts, np.zeros((3,)))
            self._center = (1.0 / len(verts)) * center_pt
        return self._center

    def normal(self) -> np.ndarray:
        if self.geometry is None or True:
            return np.zeros((3,))
        if self._normal is None:
            breakpoint()
        return self._normal

    def hull(self) -> components.Hull:
        verts = utils.GetVertices(self.geometry, assert_count=False)
        try:
            scipy_hull = ConvexHull(verts)
            return (verts, scipy_hull.simplices)
        except:
            pass

    def __eq__(self, other) -> bool:
        if not isinstance(other, CSpaceVolume):
            return False
        return self.label == other.label

    def __hash__(self) -> int:
        return hash(self.label)


@dataclass
class CSpaceGraph:
    V: List[CSpaceVolume]
    E: List[Tuple[CSpaceVolume, CSpaceVolume]]

    def label_dict(self) -> Dict[CSpaceVolume, str]:
        label_dict = dict()
        for v in self.V:
            label_dict[v] = utils.label_to_str(v.label)
        return label_dict

    def to_nx(self, start_pose: np.ndarray = None, n_closest: int = 4) -> nx.Graph:
        nx_graph = nx.Graph()
        for e in self.E:
            nx_graph.add_edge(e[0], e[1])

        if start_pose is not None:
            differences = []
            free_space = CSpaceVolume(contact_defs.fs, None)
            self.V.append(free_space)
            for v in self.V:
                differences.append(np.linalg.norm(start_pose - v.center))
            smallest_indices = np.argpartition(np.array(differences), n_closest)
            for idx in range(n_closest):
                fc_neighbor = self.V[smallest_indices[idx]]
                nx_graph.add_edge(free_space, fc_neighbor)

        return nx_graph.to_directed()

    def GetNode(self, label: components.ContactState) -> CSpaceVolume:
        for v in self.V:
            if v.label == label:
                return v

    def N(self, v) -> List[CSpaceVolume]:
        neighbors = []
        for e in self.E:
            if e[0] == v:
                neighbors.append(e[1])
            if e[1] == v:
                neighbors.append(e[0])
        return neighbors


def minkowski_sum(
    B_name: str, B: HPolyhedron, A_name: str, A: HPolyhedron
) -> CSpaceVolume:
    label = frozenset(((B_name, A_name),))
    B_vertices = utils.GetVertices(B)
    A_vertices = utils.GetVertices(A)
    volume_verts = []
    for vB_idx in range(B_vertices.shape[0]):
        for vA_idx in range(A_vertices.shape[0]):
            volume_verts.append(B_vertices[vB_idx] - A_vertices[vA_idx])
    volume_verts = np.array(volume_verts)
    volume_geometry = HPolyhedron(VPolytope(volume_verts.T))
    volume_geometry = volume_geometry.ReduceInequalities()
    return CSpaceVolume(label, volume_geometry)


class Cost:
    def __init__(self, uncertainty_dir):
        self.uncertainty_dir = uncertainty_dir

    def __call__(self, v1: CSpaceVolume, v2: CSpaceVolume, attrs) -> float:
        if v2.geometry is not None:
            n = v2.normal()
        else:
            n = v1.normal()
        projection = np.dot(n, self.uncertainty_dir) * self.uncertainty_dir
        bonus = 1e-4 * np.linalg.norm(projection)
        return 1 - bonus


def MakeWorkspaceObjectFromFaces(
    faces: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> components.WorkspaceObject:
    faces_H = dict()
    for name, Hparams in faces.items():
        suffixes = ["_bottom", "_top", "_left", "_right", "_front", "_back", "_inside"]
        is_face = any([suffix in name for suffix in suffixes])
        if is_face:
            faces_H[name] = HPolyhedron(*Hparams)
    return components.WorkspaceObject("", faces_H)


def serialize_faces(face_id_to_label, mesh):
    label_to_face_id = defaultdict(set)
    label_to_verts = defaultdict(list)
    for f, l in face_id_to_label.items():
        label_to_face_id[l].add(f)
    for l, fid_set in label_to_face_id.items():
        for f in fid_set:
            f_verts = mesh.vertices[mesh.faces[f]]
            for contact in l:
                contact_set = frozenset((contact,))
                label_to_verts[contact_set].append(f_verts)
    with open("cspace_surface.pkl", "wb") as f:
        pickle.dump(label_to_verts, f)
    return label_to_face_id


def label_face(mesh, face_id, V) -> components.ContactState:
    verts = mesh.vertices[mesh.faces[face_id]]
    sampled_pts = []
    label = None
    pt_labels = []
    for sample_idx in range(5):
        w = gen.uniform(low=0, high=1, size=3)
        w /= np.sum(w)
        pt = w[0] * verts[0] + w[1] * verts[1] + w[2] * verts[2]
        sampled_pts.append(pt)
    for pt in sampled_pts:
        pt_label = set()
        for v in V:
            if v.geometry.Scale(1.01).PointInSet(pt):
                pt_label = pt_label.union(v.label)
        pt_labels.append(pt_label)
    for pt_label in pt_labels:
        if label is None:
            label = pt_label
        else:
            label = label.intersection(pt_label)
    if len(label) == 0:
        print("invalid label")
        return None
    return frozenset(label)


def cspace_vols_to_edges(hulls: List[components.Hull], V: List[CSpaceVolume]):
    empty_graph = CSpaceGraph(V, [])
    edges = set()
    meshes = []
    for hull in hulls:
        mesh = trimesh.Trimesh(vertices=hull[0], faces=hull[1])
        if not mesh.is_volume:
            mesh.fix_normals()
            if not mesh.is_volume:
                breakpoint()
        meshes.append(mesh)
    joined_mesh = meshes[0]
    skipped_meshes = []
    for mesh in meshes[1:]:
        _joined_mesh = joined_mesh.union(mesh)
        if _joined_mesh.is_volume:
            joined_mesh = _joined_mesh
        else:
            skipped_meshes.append(mesh)
    print(f"{len(skipped_meshes)=}")
    for skipped_mesh in skipped_meshes:
        joined_mesh = joined_mesh.union(skipped_mesh)
    joined_mesh.fix_normals()
    print(f"{joined_mesh.is_volume=}")
    joined_mesh.update_faces(joined_mesh.unique_faces())
    joined_mesh = joined_mesh.process()
    _, mode_graph = graph.make_abs_graphs(V, joined_mesh)
    utils.dump_mesh(joined_mesh)
    actual_edges = list(mode_graph.edges())
    return actual_edges


def make_graph(
    manipuland: components.WorkspaceObject, env: components.WorkspaceObject
) -> CSpaceGraph:
    volumes = []
    edges = []
    for manip_face in manipuland.faces.items():
        for env_face in env.faces.items():
            vol = minkowski_sum(*env_face, *manip_face)
            volumes.append(vol)
    hulls = [v.hull() for v in volumes]
    hulls = [h for h in hulls if h is not None]
    edges = cspace_vols_to_edges(hulls, volumes)
    return CSpaceGraph(volumes, edges)


def MakeModeGraphFromFaces(
    faces_env: Dict[str, Tuple[np.ndarray, np.ndarray]],
    faces_manip: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> CSpaceGraph:
    B = MakeWorkspaceObjectFromFaces(faces_env)
    A = MakeWorkspaceObjectFromFaces(faces_manip)
    graph = make_graph(A, B)
    # render_graph(updated_graph)
    return graph


def make_task_plan(
    mode_graph: CSpaceGraph,
    start_mode: components.ContactState,
    goal_mode: components.ContactState,
    uncertainty_dir: np.ndarray,
    start_pose: np.ndarray = None,
) -> List[components.ContactState]:
    G = mode_graph.to_nx(start_pose)
    h = Cost(uncertainty_dir)
    start_vtx = mode_graph.GetNode(start_mode)
    goal_vtx = mode_graph.GetNode(goal_mode)
    assert (start_vtx is not None) and (goal_vtx is not None)
    tp_vertices = nx.shortest_path(G, source=start_vtx, target=goal_vtx, weight=h)
    # normals = [tp_vtx.normal() for tp_vtx in tp_vertices]
    tp = [tp_vtx.label for tp_vtx in tp_vertices]
    return tp
