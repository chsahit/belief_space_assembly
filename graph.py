import itertools
import pickle
from collections import defaultdict
from typing import DefaultDict, List, Set

import networkx as nx
import numpy as np
import trimesh

import puzzle_contact_defs

VertexLabel = DefaultDict[int, Set["CSpaceVolume"]]


def label_vertices_by_face(c_verts: List[np.ndarray], c_faces, V: List["CSpaceVolume"]):
    label_mapping = defaultdict(set)
    for v in V:
        scaled_v = v.geometry.Scale(1.01)
        for face_id in range(len(c_faces)):
            p1_idx = c_faces[face_id][0]
            p1 = c_verts[p1_idx]
            p2_idx = c_faces[face_id][1]
            p2 = c_verts[p2_idx]
            p3_idx = c_faces[face_id][2]
            p3 = c_verts[p3_idx]
            if (
                scaled_v.PointInSet(p1)
                and scaled_v.PointInSet(p2)
                and scaled_v.PointInSet(p3)
            ):
                label_mapping[p1_idx].add(v)
                label_mapping[p2_idx].add(v)
                label_mapping[p3_idx].add(v)
    return label_mapping


def label_vertices(pts: List[np.ndarray], V: List["CSpaceVolume"]) -> VertexLabel:
    """
    returns a label L s.t.
    L[p] = {C_1, ..., C_N} s.t mesh.vertices[p] in C_n
    """
    pt_counters = dict()  # pc[V.label] = # pts
    label_mapping = defaultdict(set)
    for v in V:
        scaled_v = v.geometry.Scale(1.1)
        for i, pt in enumerate(pts):
            if scaled_v.PointInSet(pt):
                label_mapping[i].add(v)
                pt_counters[v] = pt_counters.get(v, 0) + 1
    filtered_vertices = []
    for vol, count in pt_counters.items():
        if count <= 2:
            filtered_vertices.append(vol)
    for filtered in filtered_vertices:
        V.remove(filtered)
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
    labels = label_vertices_by_face(mesh.vertices, mesh.faces, V)
    mode_graph = nx.Graph()
    for vtx_id, vols in labels.items():
        for v1, v2 in itertools.combinations(vols, 2):
            if v1 in V and v2 in V:
                mode_graph.add_edge(v1, v2)

    return mode_graph
