from typing import Dict

from pydrake.all import RigidTransform


def annotate_peg() -> Dict[str, RigidTransform]:
    epsilon = 1e-5
    id_to_rt = dict()
    for i, x in enumerate([-0.03, 0.03]):
        for j, y in enumerate([-0.03, 0.03]):
            for k, z in enumerate([-0.075, 0.075, 0.05]):
                rt = RigidTransform([x, y, z])
                name = str(i) + str(j) + str(k)
                id_to_rt["block::" + name] = rt
    return id_to_rt


def annotate_puzzle() -> Dict[str, RigidTransform]:
    epsilon = 1e-5
    id_to_rt = dict()
    sphere_xs = [-0.01]
    sphere_ys = [-0.03, 0.03]
    sphere_zs = [-0.09]
    for i, x in enumerate(sphere_xs):
        for j, y in enumerate(sphere_ys):
            for k, z in enumerate(sphere_zs):
                rt = RigidTransform([x, y, z])
                name = str(i) + str(j) + str(k)
                id_to_rt["block::" + name] = rt
    return id_to_rt


def annotate(geom: str) -> Dict[str, RigidTransform]:
    if geom == "assets/peg.urdf":
        return annotate_peg()
    elif geom == "assets/moving_puzzle.sdf":
        return annotate_puzzle()
    else:
        raise NotImplementedError
