import itertools
import pickle
from collections import defaultdict
from typing import DefaultDict, List, Set

import networkx as nx
import numpy as np
import trimesh

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


def dump_sampler_data(mesh: trimesh.Trimesh, labels: VertexLabel):
    sampler_map = defaultdict(list)
    for vtx_id, volume_set in labels.items():
        for volume in volume_set:
            sampler_map[volume.label].append(mesh.vertices[vtx_id])
    with open("cspace_surface.pkl", "wb") as f:
        pickle.dump(sampler_map, f)
    return None


def make_mode_graph(V, mesh: trimesh.Trimesh):
    # make contact graph
    # V = CSpaceVolumeGraph, E = (V1, V2) iff \exists v\in V1 and v\in V2
    labels = label_vertices(mesh.vertices, V)
    mode_graph = nx.Graph()
    for vtx_id, vols in labels.items():
        for v1, v2 in itertools.combinations(vols, 2):
            mode_graph.add_edge(v1, v2)

    return mode_graph
