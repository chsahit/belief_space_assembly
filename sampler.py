from typing import List

import numpy as np
import trimesh
from pydrake.all import HPolyhedron, RigidTransform
from trimesh import sample

import components
import cspace
import state

gen = np.random.default_rng(0)
pwx = [0.47, 0.525]
pwy = [-0.025, 0.025]


def in_workspace(pt):
    x_sat = pt[0] > pwx[0] and pt[0] > pwx[1]
    y_sat = pt[1] > pwy[0] and pt[1] < pwy[1]
    return x_sat and y_sat


def sample_from_contact(
    p: state.Particle,
    contact_des: components.ContactState,
    num_samples: int,
    mesh: trimesh.Trimesh = None,
    num_noise: int = 0,
    seed: int = -1,
    aligned: bool = False,
) -> List[RigidTransform]:
    if seed < 0:
        seed = gen.integers(100)
    if not aligned:
        rotation = p.X_WM.rotation()
    else:
        rotation = RigidTransform().rotation()
    if mesh is None:
        p.cspace_repr = cspace.ConstructCspaceSlice(
            cspace.ConstructEnv(p), rotation
        ).mesh
        mesh = p.cspace_repr
    # utils.dump_mesh(mesh)
    satisfiying_samples = []
    ef_name = list(contact_des)[0][0]
    mf_name = list(contact_des)[0][1]
    manip_face = cspace.tf_hrepr(p._manip_poly[mf_name], rotation)
    env_face = HPolyhedron(*p.constraints[ef_name])
    volume_desired = cspace.minkowski_difference(env_face, manip_face)
    scaled_vol = HPolyhedron(volume_desired).Scale(1.01)
    # scaled_vol_large = HPolyhedron(volume_desired).Scale(1.25)
    goal_loc = np.array([0, 0, 0])
    curr_dist = np.linalg.norm(p.X_WM.translation() - goal_loc)
    attempts = 0
    while len(satisfiying_samples) < num_samples:
        pt = np.array(
            sample.sample_surface(
                mesh, 1, face_weight=[1] * len(mesh.faces), seed=attempts + seed
            )[0][0],
        )
        pt_dist = np.linalg.norm(pt - goal_loc)
        improvement = (pt_dist <= curr_dist) or True
        if (volume_desired.PointInSet(pt) or scaled_vol.PointInSet(pt)) and improvement:
            satisfiying_samples.append(pt)
        attempts += 1
        if attempts > 5000 and len(satisfiying_samples) < (num_samples / 2.0):
            if not aligned:
                print("using aligned sampler")
                return sample_from_contact(
                    p,
                    contact_des,
                    num_samples,
                    num_noise=num_noise,
                    seed=seed,
                    aligned=True,
                )
            else:
                print(f"sampler failed to find targets for {contact_des}")
                return []

    for i in range(num_noise):
        t_vel = gen.uniform(low=-0.01, high=0.01, size=3)
        noised_pt = satisfiying_samples[i] + t_vel
        if mesh.contains(np.array([noised_pt]))[0]:
            satisfiying_samples[i] = noised_pt
    satisfiying_gripper_poses = []
    for pt in satisfiying_samples:
        X_WM_des = RigidTransform(rotation, pt)
        X_WG_des = X_WM_des.multiply(p.X_GM.inverse())
        satisfiying_gripper_poses.append(X_WG_des)
    return satisfiying_gripper_poses
