from typing import List

import numpy as np
import trimesh
from pydrake.all import HPolyhedron, RigidTransform
from trimesh import sample

import components
import cspace
import state
import utils

gen = np.random.default_rng(0)


def sample_from_contact(
    p: state.Particle,
    contact_des: components.ContactState,
    num_samples: int,
    mesh: trimesh.Trimesh = None,
    num_noise: int = 0,
) -> List[RigidTransform]:
    if mesh is None:
        mesh = cspace.MakeTrimeshRepr(p.X_WM.rotation(), p.constraints, p._manip_poly)
    utils.dump_mesh(mesh)
    satisfiying_samples = []
    ef_name = list(contact_des)[0][0]
    mf_name = list(contact_des)[0][1]
    env_face = HPolyhedron(*p.constraints[ef_name])
    manip_face = HPolyhedron(*p._manip_poly[mf_name])
    volume_desired = cspace.minkowski_sum(
        ef_name, env_face, mf_name, manip_face
    ).geometry.Scale(1.01)
    attempts = 0
    while len(satisfiying_samples) < num_samples:
        pt = np.array(sample.sample_surface(mesh, 1)[0][0])
        if volume_desired.PointInSet(pt):
            satisfiying_samples.append(pt)
        attempts += 1
        if attempts > 3000:
            volume_desired = volume_desired.Scale(2.0)
    for i in range(num_noise):
        t_vel = gen.uniform(low=-0.01, high=0.01, size=3)
        noised_pt = satisfiying_samples[i] + t_vel
        if mesh.contains(np.array([noised_pt]))[0]:
            satisfiying_samples[i] = noised_pt
    satisfiying_gripper_poses = []
    for pt in satisfiying_samples:
        X_WM_des = RigidTransform(p.X_WM.rotation(), pt)
        X_WG_des = X_WM_des.multiply(p.X_GM.inverse())
        satisfiying_gripper_poses.append(X_WG_des)
    return satisfiying_gripper_poses
