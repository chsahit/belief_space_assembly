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

top_touch = set((("fixed_puzzle::b3", "block::201"),))
ft = set(
    (
        ("fixed_puzzle::b3", "block::300"),
        ("fixed_puzzle::b3", "block::302"),
    )
)


def init(X_GM_x: float = 0.0) -> state.Particle:
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.3], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.09], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.01], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/fixed_puzzle.sdf", "assets/moving_puzzle.sdf", mu=0.6
    )
    return p0


def step_one() -> state.Belief:
    p_a = init(X_GM_x=-0.005)
    p_b = init(X_GM_x=0.005)
    b0 = state.Belief([p_a, p_b])
    u = search.refine_schedule(b0, top_touch, [top_touch])
    posterior = dynamics.f_bel(b0, u)
    return posterior


def color_space(
    b: state.Belief,
    CF_d: components.ContactState,
    K_star: np.ndarray,
    X_GC: RigidTransform,
):
    print("generating coloring...")
    noms = [ik_solver.project_manipuland_to_contacts(p, CF_d) for p in b.particles]
    color_dat = []
    for i, nominal in enumerate(noms):
        print("exploring neighborhood around particle ", i)
        displacements = [directed_msets.alpha(nominal) for i in range(8)]
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
        color_dat.append((differences, u_dat))
    return color_dat


def build_colormap_ft():
    b = step_one()


def build_colormap_tt():
    p_a = init(X_GM_x=-0.005)
    p_b = init(X_GM_x=0.005)
    b0 = state.Belief([p_a, p_b])
    K_star = components.stiff
    color_dat = color_space(b0, top_touch, K_star, RigidTransform())
    # print(color_dat)
    visualize_colormap(color_dat)


def rt_to_r6(X: RigidTransform):
    r = mr.so3ToVec(mr.MatrixLog3(X.rotation().matrix()))
    return np.concatenate((r, X.translation()))


def visualize_colormap(colordat):
    cmap = {1: "r", 0: "b"}
    pca = PCA(n_components=2)
    all_differences = []
    for differences, _ in colordat:
        for X_WG in differences:
            all_differences.append(rt_to_r6(X_WG))
    all_differences = np.array(all_differences)
    tfs = pca.fit_transform(all_differences)

    for differences, u_dat in colordat:
        for k, v in u_dat.items():
            if len(v) == 2:
                c = "purple"
            else:
                c = cmap[v[0]]
            pos = pca.transform(np.array([rt_to_r6(differences[k])]))
            plt.scatter(pos[0][0], pos[0][1], c=c)
    plt.savefig("colormap.png")


if __name__ == "__main__":
    build_colormap_tt()
