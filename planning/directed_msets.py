from typing import List

import numpy as np
from dataclasses import dataclass

import components
import state
from planning import motion_sets


@dataclass
class TreeNode:
    u: components.CompliantMotion
    u_pred: components.CompliantMotion
    score: int


Tree = List[TreeNode]


def alpha() -> np.ndarray:
    pass


def nearest(T: Tree, sample: np.ndarray) -> TreeNode:
    pass


def stopping_configuration(
    q_n: TreeNode, sample: np.ndarray
) -> components.CompliantMotion:
    pass


def n_rrt(
    b: state.belief,
    K_star: np.ndarray,
    X_GC: RigidTransform,
    CF_d: components.ContactState,
) -> components.CompliantMotion:
    forest = []
    for p in b.particles:
        root_u_X_WG = motion_sets.find_nearest_valid_target(p, CF_d)
        root_u_X_WC = root_u_X_WG.multiply(X_GC)
        root_u = components.CompliantMotion(X_GC, root_u_X_WC, K_star)
        # TODO: score
        score = None
        T_p = [TreeNode(u, None, score)]
        forest.append(T_p)
