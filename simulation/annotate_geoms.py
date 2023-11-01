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
    names = ["block::000", "block::100", "block::101"]
    sphere_poses = [[0.0, 0.0, -0.1], [0.001, 0.0, -0.048], [-0.001, 0.0, -0.052]]
    for name, p_MS in zip(names, sphere_poses):
        rt = RigidTransform(p_MS)
        id_to_rt[name] = rt
    return id_to_rt


def annotate(geom: str) -> Dict[str, RigidTransform]:
    if geom == "assets/peg.urdf":
        return annotate_peg()
    elif geom == "assets/moving_puzzle.sdf":
        return annotate_puzzle()
    else:
        raise NotImplementedError
