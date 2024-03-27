import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import trimesh
from pydrake.all import HPolyhedron

import components

gen = np.random.default_rng(1)

Hull = Tuple[List[np.ndarray], List[List[float]]]
Face = Tuple[int, ...]
Facet = Tuple[int, ...]


def make_face_facet_mapping(mesh: trimesh.Trimesh) -> Dict[Face, Facet]:
    face_mapping = dict()
    for i, facet in enumerate(mesh.facets):
        for face_idx in range(facet.shape[0]):
            face = facet[face_idx]
            assert face not in face_mapping.keys()
            face_mapping[face] = i
    return face_mapping


def label_facets(mesh: trimesh.Trimesh, V) -> Dict[int, components.ContactState]:
    label_dict = dict()
    for i, facet in enumerate(mesh.facets):
        sampled_pts = []
        intersection = None
        for face_idx in range(facet.shape[0]):
            face = facet[face_idx]
            verts = mesh.vertices[mesh.faces[face]]
            for sample_idx in range(3):
                w = gen.uniform(low=0, high=1, size=3)
                w /= np.sum(w)
                # equiv to np.dot(w, verts) I think?
                pt = w[0] * verts[0] + w[1] * verts[1] + w[2] * verts[2]
                sampled_pts.append(pt)
        for pt in sampled_pts:
            pt_contact = set()
            for v in V:
                if v.geometry[0].Scale(1).PointInSet(pt):
                    pt_contact = pt_contact.union(v.label)
            if intersection is None:
                intersection = pt_contact
            else:
                intersection = intersection.intersection(pt_contact)
        label_dict[i] = intersection
    # TODO: I don't think the goal state is in here :(
    return label_dict


def _label_face(mesh, face_id, V) -> components.ContactState:
    verts = mesh.vertices[mesh.faces[face_id]]
    sampled_pts = []
    label = set()
    for sample_idx in range(6):
        w = gen.uniform(low=0, high=1, size=3)
        w /= np.sum(w)
        pt = w[0] * verts[0] + w[1] * verts[1] + w[2] * verts[2]
        sampled_pts.append(pt)
    for pt in sampled_pts:
        for v in V:
            if v.geometry[0].PointInSet(pt):
                label = label.union(v.label)
    if len(label) > 0:
        return frozenset(label)
    for pt in sampled_pts:
        for v in V:
            if v.geometry[0].Scale(1.01).PointInSet(pt):
                label = label.union(v.label)
    assert (len(label)) > 0
    return frozenset(label)


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
            if v.geometry[0].Scale(1.01).PointInSet(pt):
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
        # return _label_face(mesh, face_id, V)
        # return frozenset((("invalid", "invalid"), ))
    return frozenset(label)


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


def cspace_vols_to_graph(hulls: List[Hull], V):
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
    print("dumping obj")
    joined_mesh_obj = joined_mesh.export(file_type="obj")
    with open("cspace.obj", "w") as f:
        f.write(joined_mesh_obj)
    print(f"{joined_mesh.triangles.shape=}")
    face_id_to_label = dict()
    label_to_neighbors = defaultdict(set)
    for face in range(joined_mesh.faces.shape[0]):
        candidate_label = label_face(joined_mesh, face, V)
        if candidate_label is not None:
            face_id_to_label[face] = candidate_label
    serialize_faces(face_id_to_label, joined_mesh)
    for pair_idx in range(face_adjacency.shape[0]):
        f0 = face_adjacency[pair_idx][0]
        f1 = face_adjacency[pair_idx][1]
        if (f0 not in face_id_to_label.keys()) or (f1 not in face_id_to_label.keys()):
            continue
        l0 = face_id_to_label[f0]
        l1 = face_id_to_label[f1]
        if l0 != l1:
            label_to_neighbors[l0].add(l1)
            label_to_neighbors[l1].add(l0)
    """
    for k, v in label_to_neighbors.items():
        print(f"{k=},")
        print(f"{v=}")
    """
    return label_to_neighbors
    # breakpoint()
