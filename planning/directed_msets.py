from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
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
    r_vel = gen.uniform(low=-0.05, high=0.05, size=3)
    t_vel = gen.uniform(low=-0.005, high=0.005, size=3)
    random_vel = np.concatenate((r_vel, t_vel))
    tf = mr.MatrixExp6(mr.VecTose3(random_vel))
    sample = RigidTransform(nominal.GetAsMatrix4() @ tf)
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
    p: state.Particle,
) -> components.CompliantMotion:
    diff = q_n.u.X_WCd.InvertAndCompose(sample)
    S = 0.25 * np.eye(4)
    S[3, 3] = 1
    diff = RigidTransform(S @ diff.GetAsMatrix4())
    stopping_X_WCd = q_n.u.X_WCd.multiply(diff)
    stopping = components.CompliantMotion(q_n.u.X_GC, stopping_X_WCd, q_n.u.K)
    return stopping


def certified_stopping_conf(
    q_ns: List[TreeNode],
    samples: List[RigidTransform],
    CF_d: components.ContactState,
    p: state.Particle,
) -> Tuple[TreeNode, components.CompliantMotion]:
    assert len(q_ns) == len(samples)
    if len(q_ns) == 1:
        stopping = stopping_configuration(q_ns[0], samples[0], p)
        if dynamics.simulate(p, stopping).satisfies_contact(CF_d):
            return q_ns[0], stopping
        return None, None
    U_stopping = [
        stopping_configuration(q_ns[i], samples[i], p) for i in range(len(q_ns))
    ]
    runs = dynamics.f_cspace(p, U_stopping)
    for i, run in enumerate(runs):
        if run.satisfies_contact(CF_d):
            return q_ns[i], U_stopping[i]
    return None, None


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
        print(f"init tree with {root_score=}")
        if root_score == len(b.particles):
            soln = root_u
        root = TreeNode(root_u, None, root_score)
        forest.append(Tree(p, [root]))

    return forest, soln


def grow_tree_to_sample(
    b: state.Belief,
    T: Tree,
    samples: List[RigidTransform],
    CF_d: components.ContactState,
) -> Tuple[TreeNode, Optional[components.CompliantMotion]]:
    q_nears = [nearest(T, sample) for sample in samples]
    q_near, u_new = certified_stopping_conf(q_nears, samples, CF_d, T.p)
    if u_new is not None:
        u_new_score = score_motion(b, u_new, CF_d)
        print(f"adding node with score: {u_new_score}")
        if u_new_score == len(b.particles):
            return q_near, u_new
        q_new = TreeNode(u_new, q_near, u_new_score)
        T.nodes.append(q_new)
        return q_near, None
    else:
        return None, None


def n_rrt(
    X_GC: RigidTransform,
    K_star: np.ndarray,
    b: state.Belief,
    CF_d: components.ContactState,
) -> components.CompliantMotion:
    print("init trees")
    forest, u_star = init_trees(b, K_star, X_GC, CF_d)
    if u_star is not None:
        return u_star
    curr_tree_idx = 0
    max_iters = 100
    for iter in range(max_iters):
        print(f"propose node for tree {curr_tree_idx}")
        T_curr = forest[curr_tree_idx]
        nominal = forest[curr_tree_idx].nodes[0].u.X_WCd
        samples = [alpha(nominal) for i in range(8)]
        q_new, u_star = grow_tree_to_sample(b, T_curr, samples, CF_d)
        if u_star is not None:
            return u_star
        if q_new is not None:
            for idx, T in enumerate(forest):
                if idx == curr_tree_idx:
                    continue
                print(f"check node on tree {idx}")
                _, u_star = grow_tree_to_sample(b, T, [q_new.u.X_WCd], CF_d)
                if u_star is not None:
                    return u_star
        tree_sizes = [len(T.nodes) for T in forest]
        print(f"{tree_sizes=}")
        curr_tree_idx = tree_sizes.index(min(tree_sizes))
    return None
