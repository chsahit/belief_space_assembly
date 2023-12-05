import sys
import time
from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import RigidTransform
from sklearn.decomposition import PCA

import components
import dynamics
import mr
import state
import utils
from planning import directed_msets, search
from simulation import ik_solver

top_touch = set((("big_fixed_puzzle::b3", "block::201"),))
ft = set(
    (
        ("big_fixed_puzzle::b3", "block::300"),
        ("big_fixed_puzzle::b3", "block::302"),
    )
)


def init(X_GM_x: float = 0.0) -> state.Particle:
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.09], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.01], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/big_fixed_puzzle.sdf", "assets/moving_puzzle.sdf", mu=0.6
    )
    return p0


def step_one() -> state.Belief:
    p_a = init(X_GM_x=-0.005)
    p_b = init(X_GM_x=0.005)
    b0 = state.Belief([p_a, p_b])
    u = search.refine_schedule(b0, top_touch, [top_touch])[0]
    posterior = dynamics.f_bel(b0, u)
    return posterior


def color_space(
    b: state.Belief,
    CF_d: components.ContactState,
    K_star: np.ndarray,
    X_GC: RigidTransform,
    num_samples: int,
):
    print(f"generating coloring with {num_samples * len(b.particles)} samples")
    start_time = time.time()
    noms = [ik_solver.project_manipuland_to_contacts(p, CF_d) for p in b.particles]
    color_dat = []
    for i, nominal in enumerate(noms):
        print("exploring neighborhood around particle ", i)
        displacements = [directed_msets.alpha(nominal) for i in range(num_samples)]
        displacements[0] = nominal
        differences = [
            noms[0].InvertAndCompose(displacement) for displacement in displacements
        ]
        motions = [
            components.CompliantMotion(X_GC, displacement, K_star)
            for displacement in displacements
        ]
        u_dat = defaultdict(list)
        for p_idx, p in enumerate(b.particles):
            results = dynamics.f_cspace(p, motions)
            successes = [result.satisfies_contact(CF_d) for result in results]
            for i, succ in enumerate(successes):
                if succ:
                    u_dat[i].append(p_idx)
        for i in range(len(motions)):
            if len(u_dat[i]) == 0:
                u_dat[i].append(-1)
        color_dat.append((differences, u_dat))
    print(f"finished coloring in {time.time() - start_time} seconds")
    return color_dat


def build_colormap_ft(num_samples: int):
    print("initializing belief state...")
    b = step_one()
    K_star = components.stiff
    color_dat = color_space(b, front_touch, K_star, RigidTransform(), num_samples)
    visualize_colormap(color_dat, "colormap_ft")


def build_colormap_tt(num_samples: int):
    p_a = init(X_GM_x=-0.005)
    p_b = init(X_GM_x=0.005)
    b0 = state.Belief([p_a, p_b])
    K_star = components.stiff
    color_dat = color_space(b0, top_touch, K_star, RigidTransform(), num_samples)
    # print(color_dat)
    visualize_colormap(color_dat, "colormap_tt")


def rt_to_r6(X: RigidTransform):
    r = mr.so3ToVec(mr.MatrixLog3(X.rotation().matrix()))
    return np.concatenate((r, X.translation()))


def visualize_colormap(colordat, fname: str):
    cmap = {1: "r", 0: "b"}
    total_successes = 0
    succ_map = {0: 0, 1: 0}
    pca = PCA(n_components=2)
    all_differences = []
    for differences, _ in colordat:
        for X_WG in differences:
            all_differences.append(rt_to_r6(X_WG))
    all_differences = np.array(all_differences)
    tfs = pca.fit_transform(all_differences)

    black_circles = []
    colored_circles = []
    purple_circles = []

    for differences, u_dat in colordat:
        for k, v in u_dat.items():
            pos = pca.transform(np.array([rt_to_r6(differences[k])]))
            if len(v) == 2:
                purple_circles.append((pos, "purple"))
                total_successes += 1
            elif v[0] == -1:
                black_circles.append((pos, "black"))
            else:
                colored_circles.append((pos, cmap[v[0]]))
                succ_map[v[0]] += 1

    print(f"{total_successes=}, {succ_map=}")
    all_circles = black_circles + colored_circles + purple_circles
    for pos, c in all_circles:
        plt.scatter(pos[0][0], pos[0][1], c=c)

    plt.savefig(fname + "_bg.png")
    plt.clf()
    all_circles = colored_circles + purple_circles
    for pos, c in all_circles:
        plt.scatter(pos[0][0], pos[0][1], c=c)
    plt.savefig(fname + ".png")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        num_samples = 5
    else:
        num_samples = int(sys.argv[1])
    build_colormap_tt(num_samples)
