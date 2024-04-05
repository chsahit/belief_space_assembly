import itertools
import pickle
from collections import defaultdict
from typing import DefaultDict, List, Set

import networkx as nx
import numpy as np
import trimesh

import visualize

VertexLabel = DefaultDict[int, Set["CSpaceVolume"]]


def label_vertices(pts: List[np.ndarray], V: List["CSpaceVolume"]) -> VertexLabel:
    """
    returns a label L s.t.
    L[p] = {C_1, ..., C_N} s.t mesh.vertices[p] in C_n
    """
    label_mapping = defaultdict(set)
    for v in V:
        scaled_v = v.geometry.Scale(1.1)
        for i, pt in enumerate(pts):
            if scaled_v.PointInSet(pt):
                label_mapping[i].add(v)
    return label_mapping


def get_facet_id(face_id: int, mesh: trimesh.Trimesh) -> int:
    facets = list()
    for facet_id, facet in enumerate(mesh.facets):
        if face_id in facet:
            facets.append(facet_id)
    if len(facets) == 0:
        return len(mesh.facets)
    assert len(facets) > 0
    if len(facets) > 1:
        breakpoint()
    return facets[0]


def dump_sampler_data(mesh: trimesh.Trimesh, labels: VertexLabel):
    sampler_map = defaultdict(list)
    for vtx_id, volume_set in labels.items():
        for volume in volume_set:
            sampler_map[volume.label].append(mesh.vertices[vtx_id])
    with open("cspace_surface.pkl", "wb") as f:
        pickle.dump(sampler_map, f)
    return None


def make_abs_graphs(V: List["CSpaceVolume"], mesh: trimesh.Trimesh):
    # make facet graph
    # V = Facets; E = (F1, F2) iff \exists (fa \in F1 and fb\in F2) s.t. (fa, fb) adjacent
    facet_graph = nx.Graph()
    for pair in mesh.face_adjacency:
        facets = (get_facet_id(pair[0], mesh), get_facet_id(pair[1], mesh))
        facet_a = min(facets)
        facet_b = max(facets)
        if facet_a != facet_b:
            facet_graph.add_edge(facet_a, facet_b)
    graph_labels = dict([(v, str(v)) for v in facet_graph.nodes()])
    visualize.render_graph(facet_graph, graph_labels)
    # make contact graph
    # V = CSpaceVolumeGraph, E = (V1, V2) iff \exists v\in V1 and v\in V2
    labels = label_vertices(mesh.vertices, V)
    dump_sampler_data(mesh, labels)
    mode_graph = nx.Graph()
    for vtx_id, vols in labels.items():
        for v1, v2 in itertools.combinations(vols, 2):
            mode_graph.add_edge(v1, v2)

    # also return serializable representation
    # mapping[csv_label] = List[np.ndarray], maybe mapping[facet][CSV_label] = List[np.ndarray]
    print("warning, nothing is being serialized")
    return facet_graph, mode_graph
