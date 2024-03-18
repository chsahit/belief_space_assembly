import itertools
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cdd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pydrake.all import (
    HPolyhedron,
    Intersection,
    RandomGenerator,
    RigidTransform,
    VPolytope,
)
from scipy.spatial import ConvexHull
from tqdm import tqdm

import components
import contact_defs
import puzzle_contact_defs
import state
from simulation import hyperrectangle

drake_rng = RandomGenerator(0)


@dataclass
class WorkspaceObject:
    name: str
    geometry: HPolyhedron
    faces: Dict[str, HPolyhedron]


def largest_normal(H: HPolyhedron) -> np.ndarray:
    vertices = GetVertices(H, assert_count=False)
    try:
        hull = ConvexHull(vertices)
        simplices = hull.simplices
        assert len(simplices) > 0
    except:
        print("warning, couldn't compute simplical facets")
        return np.array([0, 0, 1])
    biggest_triangle = None
    biggest_vol = float("-inf")
    for s in simplices:
        triangle = vertices[s]
        ab = triangle[1] - triangle[0]
        ac = triangle[2] - triangle[0]
        vol = 0.5 * np.linalg.norm(np.cross(ab, ac))
        if vol > biggest_vol:
            biggest_triangle = triangle
            biggest_vol = vol
    n = np.cross(
        biggest_triangle[1] - biggest_triangle[0],
        biggest_triangle[2] - biggest_triangle[0],
    )
    return n / np.linalg.norm(n)


@dataclass
class CSpaceVolume:
    label: components.ContactState
    geometry: List[HPolyhedron]
    reached: bool = False

    def intersects(self, other: "CSpaceVolume") -> bool:
        for m_geom in self.geometry:
            for o_geom in other.geometry:
                if m_geom.IntersectsWith(o_geom):
                    return True
        return False

    def sample(self) -> np.ndarray:
        H = random.choice(self.geometry)
        return H.UniformSample(drake_rng, mixing_steps=1000)

    def is_strict_interior(self, pt) -> bool:
        # if point is exterior to all geometries, it cannot be strictly interior
        coarse_exterior_check = [not geom.PointInSet(pt) for geom in self.geometry]
        if all(coarse_exterior_check):
            return False
        # if point is strictly interior to any geometry, return true
        results = defaultdict(list)
        for direction in [0, 1, 2]:
            for delta in [0.0001, -0.0001]:
                pt_prime = np.copy(pt)
                pt_prime[direction] += delta
                for geom in self.geometry:
                    results[(direction, delta)].append(geom.PointInSet(pt_prime))
        for k, v in results.items():
            # we found a delta such that pt + delta is exterior
            # thus, the pt is a boundary pt
            if not any(v):
                return False
        return True

    def volume(self) -> float:
        _, bounds = hyperrectangle.CalcAxisAlignedBoundingBox(self.geometry[0])

        vol = (
            abs(bounds[0][0] - bounds[1][0])
            * abs(bounds[0][1] - bounds[1][1])
            * abs(bounds[0][2] - bounds[1][2])
        )
        if vol == 0:
            vol += 1e-9
        return vol

    def _normal_candidate(self, sample, A, b) -> np.ndarray:
        s_homog = np.concatenate((sample, np.array([1])))
        lb = float("inf")
        n_cand = None
        for row in range(A.shape[0]):
            plane_eqn = np.concatenate((A[row], np.array([-b[row]])))
            dist = abs(np.dot(plane_eqn, s_homog)) / np.linalg.norm(A[row])
            if dist < lb:
                lb = dist
                n_cand = A[row] / np.linalg.norm(A[row])
        assert n_cand is not None
        return n_cand

    def _modal_vec(self, cands: List[np.ndarray]) -> np.ndarray:
        scores = []
        for cand_1 in cands:
            c_scores = []
            for cand_2 in cands:
                v1 = cand_1 / np.linalg.norm(cand_1)
                v2 = cand_2 / np.linalg.norm(cand_2)
                c_scores.append(abs(np.dot(v1, v2)))
            scores.append(sum(c_scores) / float(len(c_scores)))
        highest_score = max(scores)
        return cands[scores.index(highest_score)]

    def normal(self) -> np.ndarray:
        assert len(self.geometry) > 0 or "free" in str(self.label)
        if len(self.geometry) == 0:
            return np.zeros((3,))
        return largest_normal(self.geometry[0])
        n_hat = np.array([0.0, 0.0, 0.0])
        A = self.geometry[0].A()
        b = self.geometry[0].b()
        samples = [self.sample() for i in range(10)]
        n_cands = [self._normal_candidate(sample, A, b) for sample in samples]
        return self._modal_vec(n_cands)

    def __eq__(self, other) -> bool:
        if not isinstance(other, CSpaceVolume):
            return False
        return self.label == other.label

    def __hash__(self) -> int:
        return hash(self.label)


def _g(v1: CSpaceVolume, v2: CSpaceVolume, attrs) -> float:
    new_contact_penalty = 0e-0 * int(not v2.reached)
    low_measure_penalty = 1e-4 * (1.0 / v2.volume())
    return (
        np.linalg.norm(v1.sample() - v2.sample())
        + low_measure_penalty
        + new_contact_penalty
    )


def g(v1: CSpaceVolume, v2: CSpaceVolume, uncertainty_dir: np.ndarray) -> float:
    if len(v2.geometry) > 0:
        n = v2.normal()
    else:
        n = v1.normal()
    projection = np.dot(n, uncertainty_dir) * uncertainty_dir
    bonus = 1e-4 * np.linalg.norm(projection)
    return 1 - bonus


class Cost:
    def __init__(self, uncertainty_dir):
        self.uncertainty_dir = uncertainty_dir

    def __call__(self, v1: CSpaceVolume, v2: CSpaceVolume, attrs) -> float:
        return g(v1, v2, self.uncertainty_dir)


@dataclass
class CSpaceGraph:
    V: List[CSpaceVolume]
    E: List[Tuple[CSpaceVolume, CSpaceVolume]]
    cache: Tuple[CSpaceVolume] = tuple()

    def __str__(self) -> str:
        edge_strs = []
        for e in self.E:
            edge_strs.append((e[0].label, e[1].label))
        return str(edge_strs)

    def to_nx(self) -> nx.Graph:
        nx_graph = nx.Graph()
        for e in self.E:
            if e[0].label in self.cache:
                e[0].reached = True
            else:
                e[0].reached = False
            if e[1].label in self.cache:
                e[1].reached = True
            else:
                e[1].reached = False
            nx_graph.add_edge(e[0], e[1])
        fc = None
        for v in self.V:
            if (
                v.label == contact_defs.chamfer_init
                or v.label == puzzle_contact_defs.top_touch
            ):
                fc = v
        nx_graph.add_edge(CSpaceVolume(contact_defs.fs, []), fc)
        return nx_graph


def GetVertices(H: HPolyhedron, assert_count: bool = True) -> np.ndarray:
    try:
        V = VPolytope(H.ReduceInequalities())
    except:
        # software engineering is my passion
        try:
            V = VPolytope(H.ReduceInequalities(tol=1e-6))
        except Exception as e:
            breakpoint()
    vertices = V.vertices().T
    if vertices.shape[0] < 6:

        def MatToArr(m: cdd.Matrix) -> np.ndarray:
            return np.array([m[i] for i in range(m.row_size)])

        def HPolyhedronToVRepr(H: HPolyhedron) -> cdd.Matrix:
            print("CDD fallback")
            A, b = (H.A(), H.b())
            H_repr = np.hstack((np.array([b]).T, -A))
            mat = cdd.Matrix(H_repr, number_type="float")
            mat.rep_type = cdd.RepType.INEQUALITY
            poly = cdd.Polyhedron(mat)
            return poly.get_generators()

        vertices = MatToArr(HPolyhedronToVRepr(H))[:, 1:]

    if assert_count:
        if vertices.shape[0] < 6:
            breakpoint()
        assert vertices.shape == (8, 3) or vertices.shape == (6, 3)
    return vertices


def TF_HPolyhedron(H: HPolyhedron, X_MMt: RigidTransform) -> HPolyhedron:
    vertices = GetVertices(H)
    tf = X_MMt.inverse().GetAsMatrix4()
    vertices_tf = []
    for v_idx in range(vertices.shape[0]):
        vert = vertices[v_idx]
        homogenous = np.array([vert[0], vert[1], vert[2], 1])
        homogenous_tf = tf @ homogenous
        vertices_tf.append(homogenous_tf[:3])
    return HPolyhedron(VPolytope(np.array(vertices_tf)))


def minkowski_sum(
    B_name: str, B: HPolyhedron, A_name: str, A: HPolyhedron
) -> CSpaceVolume:
    label = frozenset(((B_name, A_name),))
    B_vertices = GetVertices(B)
    A_vertices = GetVertices(A)
    volume_verts = []
    for vB_idx in range(B_vertices.shape[0]):
        for vA_idx in range(A_vertices.shape[0]):
            volume_verts.append(B_vertices[vB_idx] - A_vertices[vA_idx])
    volume_verts = np.array(volume_verts)
    volume_geometry = HPolyhedron(VPolytope(volume_verts.T))
    volume_geometry = volume_geometry.ReduceInequalities()
    return CSpaceVolume(label, [volume_geometry])


def MakeWorkspaceObjectFromFaces(
    faces: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> WorkspaceObject:
    faces_H = dict()
    for k, v in faces.items():
        if is_face(k):
            faces_H[k] = HPolyhedron(*v)
    return WorkspaceObject("", None, faces_H)


def MakeModeGraphFromFaces(
    faces_env: Dict[str, Tuple[np.ndarray, np.ndarray]],
    faces_manip: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> CSpaceGraph:
    B = MakeWorkspaceObjectFromFaces(faces_env)
    A = MakeWorkspaceObjectFromFaces(faces_manip)
    graph = make_graph([A], [B])
    return graph


def is_face(geom_name):
    suffixes = ["_bottom", "_top", "_left", "_right", "_front", "_back", "_inside"]

    is_badface = ("_top" in geom_name) and ("bottom_top" not in geom_name)
    is_chamfer = "chamfer" in geom_name
    return any([suffix in geom_name for suffix in suffixes])  # and (not is_badface)


def internal_edge(e: Tuple[CSpaceVolume, CSpaceVolume], cspace: CSpaceVolume) -> bool:
    assert len(e[0].geometry) == 1 and len(e[1].geometry) == 1
    intersection = e[0].geometry[0].Intersection(e[1].geometry[0])
    assert not intersection.IsEmpty()
    for i in range(15):
        try:
            samples = [
                intersection.UniformSample(drake_rng, intersection.ChebyshevCenter())
            ]
        except:
            samples = []
        if len(samples) > 0:
            break
    if len(samples) == 0:
        return False  # probably true, but we are being conservative in pruning
    for i in range(100):
        samples.append(intersection.UniformSample(drake_rng, samples[-1]))

    for sample in samples:
        # this edge corresponds to an external edge on the CObs
        if not cspace.is_strict_interior(sample):
            return False
    return True


def prune_edges(
    cspace_volumes: List[CSpaceVolume],
    E: List[Tuple[CSpaceVolume, CSpaceVolume]],
) -> List[Tuple[CSpaceVolume, CSpaceVolume]]:
    all_volume_geometries = sum([vol.geometry for vol in cspace_volumes], [])
    full_cspace = CSpaceVolume("", all_volume_geometries)
    # plotly_render(full_cspace)
    return E
    pruned_edges = []
    for e in tqdm(E):
        if not internal_edge(e, full_cspace):
            pruned_edges.append(e)
    return pruned_edges


def is_subset(H_A: HPolyhedron, H_B: HPolyhedron) -> bool:
    vertices = GetVertices(H_A)
    for v_idx in range(vertices.shape[0]):
        if not H_B.PointInSet(vertices[v_idx]):
            return False
    return True


def make_graph(
    manipuland: List[WorkspaceObject], env: List[WorkspaceObject]
) -> CSpaceGraph:
    to_prune = []
    ec = env[0]
    for env_face_A in ec.faces.items():
        for env_face_B in ec.faces.items():
            if env_face_A[0] == env_face_B[0]:
                continue
            H_A = env_face_A[1]
            H_B = env_face_B[1]
            """
            if "front_bottom" in env_face_A[0] and "bottom_top" in env_face_B[0]:
                breakpoint()
            """
            if is_subset(H_A, H_B):
                to_prune.append(env_face_A[0])
            if is_subset(H_B, H_A):
                to_prune.append(env_face_B[0])

    to_prune = set(to_prune)
    print(f"{to_prune=}")
    ec_pruned_dict = dict()
    for ef in ec.faces.items():
        if ef[0] in to_prune:
            continue
        ec_pruned_dict[ef[0]] = ef[1]
    env_pruned = [WorkspaceObject("env", [], ec_pruned_dict)]
    volumes = []
    edges = []
    for manip_component in manipuland:
        for env_component in env_pruned:
            for manip_face in manip_component.faces.items():
                for env_face in env_component.faces.items():
                    vol = minkowski_sum(*env_face, *manip_face)
                    volumes.append(vol)
    for pair in itertools.combinations(volumes, 2):
        if pair[0].intersects(pair[1]):
            edges.append(pair)
    edges = prune_edges(volumes, edges)
    return CSpaceGraph(volumes, edges)


def label_to_str(label: components.ContactState) -> str:
    tag = ""
    for contact in label:
        A = contact[0][contact[0].find("::") + 2 :]
        B = contact[1][contact[1].find("::") + 2 :]
        tag += f"({A}, {B}), "
    return tag


def render_graph(g: CSpaceGraph, plotly: bool = True):
    nx_graph = nx.Graph()
    label_dict = dict()
    for e in g.E:
        nx_graph.add_edge(e[0], e[1])
        label_dict[e[0]] = label_to_str(e[0].label)
        label_dict[e[1]] = label_to_str(e[1].label)
    print(f"{nx.number_connected_components(nx_graph)=}")
    if not plotly:
        nx.draw(nx_graph, labels=label_dict, with_labels=True)
        plt.tight_layout()
        plt.savefig("mode_graph.png", bbox_inches="tight", dpi=300)
        return
    edge_x = []
    edge_y = []
    pos = nx.spring_layout(nx_graph)
    for edge in nx_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), mode="lines"
    )
    node_x = [pos[node][0] for node in nx_graph.nodes()]
    node_y = [pos[node][1] for node in nx_graph.nodes()]
    labels = [label_dict[node] for node in nx_graph.nodes()]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            line_width=2,
        ),
    )
    node_trace.text = labels
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False)
    fig.write_html("mode_graph.html")


def render_cspace_volume(C: CSpaceVolume):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = ["r", "g", "b", "black", "y", "p"]
    for i, geom in enumerate(C.geometry):
        vertices = GetVertices(geom, assert_count=False)
        try:
            hull = ConvexHull(vertices)
            color = colors[i % 3]
            for s in hull.simplices:
                tri = Poly3DCollection([vertices[s]])
                tri.set_color(color)
                # tri.set_alpha(0.5)
                ax.add_collection3d(tri)
        except:
            pass
    plt.show()


def plotly_render(C: CSpaceVolume):
    meshes = []
    for geom in C.geometry:
        vertices = GetVertices(geom, assert_count=False)
        try:
            hull = ConvexHull(vertices).simplices.T.tolist()
        except Exception as e:
            try:
                hull = ConvexHull(vertices, qhull_options="QJn").simplices.T.tolist()
            except:
                print("ignoring a hull")
                hull = None

        vertices = vertices.T.tolist()
        if hull is not None:
            meshes.append(
                go.Mesh3d(
                    x=vertices[0],
                    y=vertices[1],
                    z=vertices[2],
                    color="lightpink",
                    opacity=1.0,
                    i=hull[0],
                    j=hull[1],
                    k=hull[2],
                )
            )

    print(f"{len(meshes)=}")
    fig = go.Figure(data=meshes)
    fig.write_html("cso.html")


def compute_uncertainty_dir(configs: List[np.ndarray]) -> np.ndarray:
    max_dist = float("-inf")
    worst_pair = None
    for p1 in configs:
        for p2 in configs:
            dist = np.linalg.norm(p1 - p2)
            if dist > max_dist:
                max_dist = dist
                worst_pair = (p1, p2)
    direction = worst_pair[0] - worst_pair[1]
    if np.linalg.norm(direction) < 1e-9:
        direction = np.array([0, 0, 1.0])
    else:
        direction = direction / np.linalg.norm(direction)
    print(f"{direction=}")
    assert abs(np.linalg.norm(direction) - 1) < 1e-4
    return direction


def make_task_plan(
    mode_graph: CSpaceGraph,
    start_mode: components.ContactState,
    goal_mode: components.ContactState,
    curr_configurations: List[np.ndarray],
) -> List[components.ContactState]:
    G = mode_graph.to_nx()
    h = Cost(compute_uncertainty_dir(curr_configurations))
    start_vtx = None
    goal_vtx = None
    for v in G.nodes():
        if v.label == start_mode:
            start_vtx = v
        if v.label == goal_mode:
            goal_vtx = v
    assert (start_vtx is not None) and (goal_vtx is not None)
    print("computing task plan")
    tp_vertices = nx.shortest_path(G, source=start_vtx, target=goal_vtx, weight=h)
    normals = [tp_vtx.normal() for tp_vtx in tp_vertices]
    print(f"{normals=}")
    tp = [tp_vtx.label for tp_vtx in tp_vertices]
    # vols = [tp_vtx.volume() for tp_vtx in tp_vertices]
    # print(f"{vols=}")
    return tp
