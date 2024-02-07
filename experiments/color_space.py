from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import state
import utils
from experiments import init_particle
from planning import refine_motion
from simulation import generate_contact_set, ik_solver


def init():
    modes = [
        contact_defs.chamfer_touch_2,
        contact_defs.front_faces,
    ]
    p0 = init_particle.init_peg(X_GM_x=-0.02)
    p1 = init_particle.init_peg(pitch=0)
    p2 = init_particle.init_peg(X_GM_x=0.02)
    b0 = state.Belief([p0, p1, p2])
    result = refine_motion.randomized_refine(b0, modes, max_attempts=3)
    b_init = b0
    for u in result.traj:
        b_init = dynamics.f_bel(b_init, u)
    return b_init


Points = List[float]
Coloring = Tuple[Points, Points, Points, List[str]]


def generate_coloring(b: state.Particle) -> Coloring:
    CF_d = contact_defs.bottom_faces_fully_constrained
    cmap = {0: "r", 1: "g", 2: "b", 3: "black"}
    nominal = utils.xyz_rpy_deg([0, 0, 0], [180, 0, 0])
    X_GC = RigidTransform([0.0, 0.0, 0.15])
    K = components.stiff

    xs = []
    zs = []
    pitches = []
    colors = []

    for i in range(len(b.particles)):
        print(f"coloring for particle {i}")
        targets = generate_contact_set.project_manipuland_to_contacts(
            b.particles[i], contact_defs.bottom_faces_fully_constrained, num_samples=128
        )
        targets = [target.multiply(X_GC) for target in targets]
        motions = [components.CompliantMotion(X_GC, target, K) for target in targets]
        motions = [ik_solver.update_motion_qd(m) for m in motions]
        posteriors = dynamics.parallel_f_bel(b, motions)
        scores = [int(p.partial_sat_score(CF_d)) for p in posteriors]
        for (u, s) in zip(motions, scores):
            colors.append(cmap[s])
            X_WGd = u.X_WCd.multiply(X_GC.inverse())
            diff = nominal.inverse().multiply(X_WGd)
            xs.append(diff.translation()[0])
            zs.append(diff.translation()[2])
            rpy_diff = diff.rotation().ToRollPitchYaw().vector()
            pitches.append(rpy_diff[1])

    return (xs, zs, pitches, colors)


def color_experiment():
    print("init...")
    b0 = init()
    print("do color:")
    results = generate_coloring(b0)
    plt.scatter(results[0], results[1], c=results[3])
    plt.show()
    plt.scatter(results[1], results[2], c=results[3])
    plt.show()
    breakpoint()


if __name__ == "__main__":
    color_experiment()
