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
import puzzle_contact_defs
import utils

drake_rng = RandomGenerator(0)
gen = np.random.default_rng(1)


def TF_HPolyhedron(H: HPolyhedron, X_MMt: RigidTransform) -> HPolyhedron:
    vertices = utils.GetVertices(H)
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

    def sample(self) -> np.ndarray:
        return self.geometry.UniformSample(drake_rng, mixing_steps=1000)

    def normal(self) -> np.ndarray:
        if self.geometry is None:
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

    def to_nx(self) -> nx.Graph:
        nx_graph = nx.Graph()
        for e in self.E:
            nx_graph.add_edge(e[0], e[1])
        fc_a = self.GetNode(contact_defs.chamfer_init)
        fc_b = self.GetNode(puzzle_contact_defs.top_touch)
        fc = fc_a or fc_b
        nx_graph.add_edge(CSpaceVolume(contact_defs.fs, None), fc)
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
        meshes.append(mesh)
    joined_mesh = meshes[0]
    for mesh in meshes[1:]:
        joined_mesh = joined_mesh.union(mesh)
    joined_mesh.fix_normals()
    joined_mesh.update_faces(joined_mesh.unique_faces())
    joined_mesh = joined_mesh.process()
    face_adjacency = joined_mesh.face_adjacency
    # utils.dump_mesh(joined_mesh)
    face_id_to_label = dict()
    label_to_neighbors = defaultdict(set)
    for face in range(joined_mesh.faces.shape[0]):
        candidate_label = label_face(joined_mesh, face, V)
        if candidate_label is not None:
            face_id_to_label[face] = candidate_label
    labels_to_face_id = serialize_faces(face_id_to_label, joined_mesh)
    # compute edges
    for pair_idx in range(face_adjacency.shape[0]):
        f0 = face_adjacency[pair_idx][0]
        f1 = face_adjacency[pair_idx][1]
        if (f0 not in face_id_to_label.keys()) or (f1 not in face_id_to_label.keys()):
            continue
        l0 = face_id_to_label[f0]
        l1 = face_id_to_label[f1]
        for contact_0 in l0:
            v0 = empty_graph.GetNode(frozenset((contact_0,)))
            v0._normal = joined_mesh.face_normals[f0]
            for contact_1 in l1:
                if contact_0 == contact_1:
                    continue
                v1 = empty_graph.GetNode(frozenset((contact_1,)))
                v1._normal = joined_mesh.face_normals[f1]
                cand_edge = (v0, v1)
                if cand_edge in edges or cand_edge[::-1] in edges:
                    continue
                edges.add(cand_edge)
    return list(edges)


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
) -> List[components.ContactState]:
    G = mode_graph.to_nx()
    h = Cost(uncertainty_dir)
    start_vtx = None
    goal_vtx = None
    for v in G.nodes():
        if v.label == start_mode:
            start_vtx = v
        if v.label == goal_mode:
            goal_vtx = v
    assert (start_vtx is not None) and (goal_vtx is not None)
    tp_vertices = nx.shortest_path(G, source=start_vtx, target=goal_vtx, weight=h)
    # normals = [tp_vtx.normal() for tp_vtx in tp_vertices]
    tp = [tp_vtx.label for tp_vtx in tp_vertices]
    return tp
