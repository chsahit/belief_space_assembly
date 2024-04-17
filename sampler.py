from typing import List

import numpy as np
import trimesh
from pydrake.all import HPolyhedron, RigidTransform
from trimesh import sample

import components
import cspace
import state

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
    satisfiying_samples = []
    ef_name = list(contact_des)[0][0]
    mf_name = list(contact_des)[0][1]
    env_face = HPolyhedron(*p.constraints[ef_name])
    manip_face = HPolyhedron(*p._manip_poly[mf_name])
    volume_desired = cspace.minkowski_sum(
        ef_name, env_face, mf_name, manip_face
    ).geometry
    while len(satisfiying_samples) < num_samples:
        pt = np.array(sample.sample_surface(mesh, 1)[0][0])
        if volume_desired.PointInSet(pt):
            satisfiying_samples.append(pt)
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
