import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import trimesh
from pydrake.all import (
    HPolyhedron,
    RandomGenerator,
    RigidTransform,
    RotationMatrix,
    VPolytope,
)
from scipy.spatial import ConvexHull
from trimesh import proximity

import components
import contact_defs
import graph
import puzzle_contact_defs
import utils

drake_rng = RandomGenerator(0)
gen = np.random.default_rng(1)
global_mesh = None


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
    hp = HPolyhedron(VPolytope(np.array(vertices_tf).T))
    return hp


class CSpaceVolume:
    def __init__(self, label: components.ContactState, geometry: HPolyhedron):
        self.label = label
        self.geometry = geometry
        self._normal = None
        self._center = None

    def sample(self) -> np.ndarray:
        return self.geometry.UniformSample(drake_rng, mixing_steps=1000)

    @property
    def center(self) -> np.ndarray:
        if self.label == contact_defs.fs:
            return None
        if self._center is None:
            verts = utils.GetVertices(self.geometry)
            if verts is None:
                return np.array([100, 100, 100])
            verts = verts.tolist()
            center_pt = sum(verts, np.zeros((3,)))
            self._center = (1.0 / len(verts)) * center_pt
        return self._center

    def normal(self) -> np.ndarray:
        if self.geometry is None:
            return np.zeros((3,))
        if self._normal is None:
            assert global_mesh is not None
            center_pt = np.array([self.center])
            _, _, triangle_id = proximity.closest_point(global_mesh, center_pt)
            self._normal = global_mesh.face_normals[triangle_id][0]
        return self._normal

    def hull(self) -> components.Hull:
        verts = utils.GetVertices(self.geometry, assert_count=False)
        if verts is None:
            print(f"no verts for {self.label}")
            return None
        try:
            scipy_hull = ConvexHull(verts)
            return (verts, scipy_hull.simplices)
        except Exception:
            pass

    def __eq__(self, other) -> bool:
        if not isinstance(other, CSpaceVolume):
            return False
        return self.label == other.label

    def __hash__(self) -> int:
        return hash(self.label)


class CSpaceGraph:
    def __init__(
        self, V: List[CSpaceVolume], E: List[Tuple[CSpaceVolume, CSpaceVolume]]
    ):
        self.V = V
        self.E = E
        free_space = CSpaceVolume(contact_defs.fs, None)
        self.V.append(free_space)

    def label_dict(self) -> Dict[CSpaceVolume, str]:
        label_dict = dict()
        for v in self.V:
            label_dict[v] = utils.label_to_str(v.label)
        return label_dict

    def to_nx(self, start_pose: np.ndarray = None, n_closest: int = 1) -> nx.Graph:
        nx_graph = nx.Graph()
        free_space = self.GetNode(contact_defs.fs)
        for e in self.E:
            nx_graph.add_edge(e[0], e[1])

        if start_pose is not None and len(self.N(free_space)) == 0:
            differences = []
            for v in self.V:
                if v.center is not None:
                    differences.append(np.linalg.norm(start_pose - v.center))
                else:
                    differences.append(float("inf"))
            smallest_indices = np.argpartition(np.array(differences), n_closest)
            for idx in range(n_closest):
                fc_neighbor = self.V[smallest_indices[idx]]
                self.E.append((free_space, fc_neighbor))
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
    faces: Dict[str, Tuple[np.ndarray, np.ndarray]], only_planes: bool = True
) -> components.WorkspaceObject:
    faces_H = dict()
    for name, Hparams in faces.items():
        suffixes = ["_bottom", "_top", "_left", "_right", "_front", "_back", "_inside"]
        is_face = any([suffix in name for suffix in suffixes])
        if not only_planes:
            is_face = not is_face
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


def cspace_vols_to_trimesh(hulls: List[components.Hull]):
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
    for skipped_mesh in skipped_meshes:
        joined_mesh = joined_mesh.union(skipped_mesh)
    # joined_mesh.fix_normals()
    # joined_mesh.update_faces(joined_mesh.unique_faces())
    # joined_mesh = joined_mesh.process()
    return joined_mesh


def cspace_vols_to_edges(hulls: List[components.Hull], V: List[CSpaceVolume]):
    joined_mesh = cspace_vols_to_trimesh(hulls)
    mode_graph = graph.make_mode_graph(V, joined_mesh)
    # utils.dump_mesh(joined_mesh)
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
    global global_mesh
    if global_mesh is None:
        # software engineering is my passion
        global_mesh = cspace_vols_to_trimesh(hulls)
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


def MakeTrimeshRepr(
    R_WM: RotationMatrix,
    env_geom: Dict[str, components.HRepr],
    manip_geom: Dict[str, components.HRepr],
) -> trimesh.Trimesh:
    # breakpoint()
    transformed_manip_poly = dict()
    for name, geom in manip_geom.items():
        transformed_geom = TF_HPolyhedron(HPolyhedron(*geom), RigidTransform(R_WM))
        transformed_manip_poly[name] = (transformed_geom.A(), transformed_geom.b())
    B = MakeWorkspaceObjectFromFaces(env_geom, only_planes=False)
    A = MakeWorkspaceObjectFromFaces(transformed_manip_poly, only_planes=False)
    volumes = []
    for manip_face in A.faces.items():
        for env_face in B.faces.items():
            vol = minkowski_sum(*env_face, *manip_face)
            volumes.append(vol)
    hulls = [v.hull() for v in volumes]
    hulls = [h for h in hulls if h is not None]
    return cspace_vols_to_trimesh(hulls)


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
