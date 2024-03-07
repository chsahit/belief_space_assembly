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
from simulation import ik_solver

gen = np.random.default_rng(1)
Bound = Tuple[float, float]
workspace = ([0.48, 0.52], [-0.02, 0.02], [-0.02, 0.32])

@dataclass(frozen=True)
class BNode:
    b: state.Belief
    pred: "Tuple[BNode, components.CompliantMotion]"

    def traj(self) -> List[components.CompliantMotion]:
        motions = []
        prev = self.pred
        while prev is not None:
            motions.append(prev[1])
            prev = prev[0]
        return motions.reverse()


class SearchTree:
    def __init__(self, kd_tree: KDTree):
        self.kd_tree = kd_tree
        self.occupancy = defaultdict(list)

    def add_bel(self, bn: BNode):
        mu = bn.b.mean().X_WM
        mu_t, mu_r = mu.translation(), mu.rotation().ToQuaternion()
        mu_r = np.array([mu_r.w(), mu_r.x(), mu_r.y(), mu_r.z()])
        if mu_r[0] < 0:
            mu_r *= -1
        mu_r7 = np.concatenate((mu_r, mu_t)).reshape(1, -1)
        _, ind = self.kd_tree.query(mu_r7)
        self.occupancy[ind.item()].append(bn)

    def sample(self) -> BNode:
        random_cell = random.choice(list(self.occupancy.keys()))
        random_node = random.choice(self.occupancy[random_cell])
        return random_node


def make_kdtree(xlims: Bound, ylims: Bound, zlims: Bound, bins: int) -> KDTree:
    # bin space of orientations s.t real component is positive
    qw = np.linspace(0.0, 1.0, num=bins)
    qx = np.linspace(-1.0, 1.0, num=bins)
    qy = np.linspace(-1.0, 1.0, num=bins)
    qz = np.linspace(-1.0, 1.0, num=bins)
    x = np.linspace(xlims[0], xlims[1], num=bins)
    y = np.linspace(ylims[0], ylims[1], num=bins)
    z = np.linspace(zlims[0], zlims[1], num=bins)
    _C = np.meshgrid(qw, qx, qy, qz, x, y, z)
    C = np.array([i.flatten() for i in _C]).T
    kdtree = KDTree(C)
    return kdtree


def sample_control(b: state.Belief) -> components.CompliantMotion:
    X_GC = RigidTransform()
    K = components.stiff
    r_vel = gen.uniform(low=-0.05, high=0.05, size=3)
    t_vel = gen.uniform(low=-0.02, high=0.02, size=3)
    vel = np.concatenate((r_vel, t_vel))
    X_CCt = RigidTransform(mr.MatrixExp6(mr.VecTose3(vel)))
    X_WCt = b.mean().X_WG.multiply(X_GC).multiply(X_CCt)
    u = components.CompliantMotion(X_GC, X_WCt, K)
    return ik_solver.update_motion_qd(u)


def is_valid(b, workspace) -> bool:
    mu_t = b.mean().X_WM.translation()
    x_valid = mu_t[0] > workspace[0][0] and mu_t[0] < workspace[0][1]
    y_valid = mu_t[1] > workspace[1][0] and mu_t[1] < workspace[1][1]
    z_valid = mu_t[1] > workspace[2][0] and mu_t[2] < workspace[2][1]
    return x_valid and y_valid and z_valid


def b_est(
    b0: state.Belief, goal: components.ContactState, timeout: float = 60.0
) -> components.PlanningResult:
    start_time = time.time()
    _tree = make_kdtree(*workspace, 5)
    tree = SearchTree(_tree)
    tree.add_bel(BNode(b0, None))
    while time.time() - start_time < timeout:
        bn = tree.sample()
        u = sample_control(bn.b)
        bn_next = BNode(dynamics.f_bel(bn.b, u), (bn, u))
        if is_valid(bn_next.b, workspace):
            if bn_next.b.satisfies_contact(goal):
                traj = bn_next.traj()
                total_time = time.time() - start_time
                print(f"{len(traj)=}")
                return components.PlanningResult(traj, total_time, 0, 0, None)
        tree.add_bel(bn_next)
    total_time = time.time() - start_time
    return components.PlanningResult(None, total_time, 0, 0, None)
