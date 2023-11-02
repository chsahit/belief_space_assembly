from typing import List, Optional, Tuple

import numpy as np
from dataclasses import dataclass
from pydrake.all import RigidTransform

import components
import dynamics
import mr
import state
import utils
from planning import motion_sets

gen = np.random.default_rng()


@dataclass(frozen=True)
class TreeNode:
    u: components.CompliantMotion
    u_pred: components.CompliantMotion
    score: int

    @property
    def r7_repr(self) -> np.ndarray:
        return utils.RigidTfToVec(self.u.X_WCd)


@dataclass
class Tree:
    p: state.Particle
    nodes: List[TreeNode]


def alpha(nominal: RigidTransform) -> RigidTransform:
    r_vel = (gen.random_sample(3) * 0.2) - 0.1
    t_vel = (gen.random_sample(3) * 0.1) - 0.05
    tf = mr.MatrixExp6(np.concatenate((r_vel, t_vel)))
    sample = RigidTransform(nominal.matrix() @ tf)
    return sample


def nearest(T: Tree, sample: RigidTransform) -> TreeNode:
    sample = utils.RigidTfToVec(sample)
    min_dist = np.inf
    min_node = None
    for node in T.nodes:
        if sample[0] * node.r7_repr[0] < 0:
            sample[:4] *= -1
        dist = np.linalg.norm(node.r7_repr - sample)
        if dist < min_dist:
            min_dist = dist
            min_node = node
    return min_node


def stopping_configuration(
    q_n: TreeNode,
    sample: RigidTransform,
    CF_d: components.ContactState,
    p: state.Particle,
) -> components.CompliantMotion:
    diff = q_n.u.X_WCd.InvertAndCompose(sample)
    S = 0.25 * np.eye(4)
    S[3, 3] = 1
    diff = RigidTransform(S @ diff.matrix())
    stopping_X_WCd = q_n.u.X_WCd.multiply(diff)
    stopping = components.CompliantMotion(q_n.u.X_GC, stopping_X_WCd, q_n.u.K)
    if dynamics.simulate(p, stopping).satisfies_contact(CF_d):
        return stopping
    return None


def score_motion(
    b: state.Belief, u: components.CompliantMotion, CF_d: components.ContactState
) -> int:
    posterior = dynamics.f_bel(b, u)
    return sum([int(p.satisfies_contact(CF_d)) for p in posterior.particles])


def init_trees(
    b: state.Belief,
    K_star: np.ndarray,
    X_GC: RigidTransform,
    CF_d: components.ContactState,
) -> Tuple[List[Tree], Optional[components.CompliantMotion]]:
    forest = []
    soln = None
    for p in b.particles:
        root_u_X_WG = motion_sets.find_nearest_valid_target(p, CF_d)
        root_u_X_WC = root_u_X_WG.multiply(X_GC)
        root_u = components.CompliantMotion(X_GC, root_u_X_WC, K_star)
        root_score = score_motion(b, root_u, CF_d)
        if root_score == len(b.particles):
            soln = root_u
        root = TreeNode(root_u, None, root_score)
        forest.append(Tree(p, [root]))

    return forest, soln


def grow_tree_to_sample(
    b: state.Belief, T: Tree, sample: np.ndarray, CF_d: components.ContactState
) -> Tuple[TreeNode, Optional[components.CompliantMotion]]:
    q_near = nearest(T, sample)
    u_new = stopping_configuration(q_near, sample, CF_d, T.p)
    if u_new is not None:
        u_new_score = score_motion(b, u_new, CF_d)
        if u_new_score == len(b.particles):
            return q_near, u_new
        q_new = TreeNode(u_new, q_near, u_new_score)
        T.nodes.append(q_new)
    return q_near, None


def n_rrt(
    b: state.Belief,
    K_star: np.ndarray,
    X_GC: RigidTransform,
    CF_d: components.ContactState,
) -> components.CompliantMotion:
    forest, u_star = init_trees(b, K_star, X_GC, CF_d)
    if u_star is not None:
        return u_star
    curr_tree_idx = 0
    max_iters = 3
    nominal = b.X_WM.multiply(X_GC)
    for iter in range(max_iters):
        T_curr = forest[curr_tree_idx]
        sample = alpha(nominal)
        q_new, u_star = grow_tree_to_sample(T_curr, sample)
        if u_star is not None:
            return u_star
        r7_new = q_new.r7_repr
        for i, T in enumerate(forest):
            if i == curr_tree_idx:
                continue
            _, u_star = grow_tree_to_sample(T, r7_new)
            if u_star is not None:
                return u_star
        tree_sizes = [len(T) for T in forest]
        curr_tree_idx = tree_sizes.index(min(tree_sizes))
    return None
