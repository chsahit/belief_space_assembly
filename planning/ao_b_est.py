import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from pydrake.all import RigidTransform
from sklearn.neighbors import KDTree

import components
import dynamics
import mr
import state

random.seed(0)
gen = np.random.default_rng(1)
Bound = Tuple[float, float]
workspace_peg = ([0.495, 0.505], [-0.0075, 0.0075], [0.02, 0.205])
workspace_puzzle = ([0.49, 0.51], [-0.01, 0.0175], [0.02, 0.205])


@dataclass(frozen=True)
class BNode:
    b: state.Belief
    pred: "Tuple[BNode, components.CompliantMotion]"

    def traj(self) -> List[components.CompliantMotion]:
        motions = []
        prev = self.pred
        while prev is not None:
            motions.append(prev[1])
            prev = prev[0].pred
        motions.reverse()
        return motions


class SearchTree:
    def __init__(self, kd_tree: KDTree):
        self.kd_tree = kd_tree
        self.occupancy = defaultdict(list)
        self.num_nodes = 0

    def add_bel(self, bn: BNode):
        mu = bn.b.mean().X_WM
        mu_t, mu_r = mu.translation(), mu.rotation().ToQuaternion()
        mu_t = bn.b.mean_translation()
        mu_r = np.array([mu_r.w(), mu_r.x(), mu_r.y(), mu_r.z()])
        if mu_r[0] < 0:
            mu_r *= -1
        # mu_r7 = np.concatenate((mu_r, mu_t)).reshape(1, -1)
        # _, ind = self.kd_tree.query(mu_r7)
        _, ind = self.kd_tree.query(mu_t.reshape(1, -1))
        self.occupancy[ind.item()].append(bn)
        self.num_nodes += 1

    def sample(self) -> BNode:
        random_cell = random.choice(list(self.occupancy.keys()))
        random_node = random.choice(self.occupancy[random_cell])
        return random_node


def make_kdtree(xlims: Bound, ylims: Bound, zlims: Bound, bins: int) -> KDTree:
    # bin space of orientations s.t real component is positive
    # qw = np.linspace(0.0, 1.0, num=int(bins / 3))
    # qx = np.linspace(-1.0, 1.0, num=int(bins / 3))
    # qy = np.linspace(-1.0, 1.0, num=int(bins / 3))
    # qz = np.linspace(-1.0, 1.0, num=int(bins / 3))
    x = np.linspace(xlims[0], xlims[1], num=bins)
    y = np.linspace(ylims[0], ylims[1], num=bins)
    z = np.linspace(zlims[0], zlims[1], num=bins)
    # _C = np.meshgrid(qw, qx, qy, qz, x, y, z)
    _C = np.meshgrid(x, y, z)
    C = np.array([i.flatten() for i in _C]).T
    kdtree = KDTree(C)
    return kdtree


def sample_control(b: state.Belief) -> components.CompliantMotion:
    X_GC = RigidTransform()
    K = components.stiff
    # r_vel = gen.uniform(low=-0.02, high=0.02, size=3)
    r_vel = np.zeros((3,))
    # t_vel = gen.uniform(low=-0.01, high=0.01, size=3)
    t_vel = np.zeros((3,))
    while np.linalg.norm(t_vel) < 1e-5:
        t_vel = gen.standard_normal(size=3)
    t_vel = 0.05 * (t_vel / np.linalg.norm(t_vel))
    vel = np.concatenate((r_vel, t_vel))
    X_CCt = RigidTransform(mr.MatrixExp6(mr.VecTose3(vel)))
    X_WCt = b.mean().X_WG.multiply(X_GC).multiply(X_CCt)
    t = gen.uniform(low=0.0, high=3.0)
    u = components.CompliantMotion(X_GC, X_WCt, K, timeout=t)
    return u


def is_valid(b, workspace) -> bool:
    mu = b.mean()
    mu_t = mu.X_WM.translation()
    x_valid = mu_t[0] > workspace[0][0] and mu_t[0] < workspace[0][1]
    y_valid = mu_t[1] > workspace[1][0] and mu_t[1] < workspace[1][1]
    z_valid = mu_t[2] > workspace[2][0] and mu_t[2] < workspace[2][1]
    del mu
    return x_valid and y_valid and z_valid


def best_node(tree: SearchTree) -> BNode:
    min_z = float("inf")
    best_node = None
    for k, v in tree.occupancy.items():
        for bn in v:
            z = bn.b.mean_translation()[2]
            if z < min_z:
                best_node = bn
                min_z = z
    print(f"{min_z=}")
    return best_node


def b_est(
    b0: state.Belief, goal: components.ContactState, timeout: float = 1200.0
) -> components.PlanningResult:
    if "puzzle" in b0.particles[0].env_geom:
        workspace = workspace_puzzle
    else:
        workspace = workspace_peg
    start_time = time.time()
    last_printed_time = start_time
    _tree = make_kdtree(*workspace, 10)
    tree = SearchTree(_tree)
    tree.add_bel(BNode(b0, None))
    num_posteriors = 0
    while time.time() - start_time < timeout:
        if (time.time() - last_printed_time) > 5:
            runtime = time.time() - start_time
            print(f"{int(runtime)=}, {num_posteriors=}", end="\r")
            last_printed_time = time.time()
        bn = tree.sample()
        u = sample_control(bn.b)
        bn_next = BNode(dynamics.f_bel(bn.b, u), (bn, u))
        num_posteriors += len(bn.b.particles)
        if is_valid(bn_next.b, workspace):
            tree.add_bel(bn_next)
            if bn_next.b.satisfies_contact(goal):
                traj = bn_next.traj()
                total_time = time.time() - start_time
                del _tree
                del tree
                # print(f"{len(traj)=}")
                # print(f"{total_time=}")
                return components.PlanningResult(traj, total_time, 0, 0, None)
    print("")
    total_time = time.time() - start_time
    best_traj = best_node(tree).traj()
    del _tree
    del tree
    # print(f"{tree.num_nodes=}, {num_posteriors=}")
    print(f"returning best non-satifiying traj, len={len(best_traj)}")
    return components.PlanningResult(best_traj, total_time, 0, 0, None)
