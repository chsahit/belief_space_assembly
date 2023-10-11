from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt

plt.switch_backend("agg")
import multiprocessing
import signal

import numpy as np
from PIL import Image
from pydrake.all import RigidTransform
from tqdm import tqdm

import components
import contact_defs
import dynamics
import state
import utils
from simulation import ik_solver

# List of tuples of form (X_AC_x, X_AC_z, score)
ExperimentResult = List[Tuple[float, float, int]]
num_particles = None


def b() -> state.Belief:
    global num_particles
    X_WG_0 = utils.xyz_rpy_deg([0.50, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([0.0, 0.0, 0.155], [0, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/big_chamfered_hole.sdf", "assets/peg.urdf", mu=0.6
    )

    grasps = [
        components.Grasp(x=0.0, z=0.155, pitch=0),
        components.Grasp(x=-0.01, z=0.155, pitch=0),
        components.Grasp(x=0.01, z=0.155, pitch=0),
        components.Grasp(x=0.0, z=0.155, pitch=-5),
        components.Grasp(x=0.0, z=0.155, pitch=5),
    ]
    bin_poses = [components.ObjectPose(x=0.5, y=0, yaw=0)] * len(grasps)
    b = state.Belief.make_particles(grasps, bin_poses, p0)
    num_particles = len(b.particles)
    return b


def map_motion_to_GC_frame(
    u_nom: components.CompliantMotion, X_GC: RigidTransform
) -> components.CompliantMotion:
    return components.CompliantMotion(
        X_GC, u_nom.X_WCd.multiply(X_GC), u_nom.K, timeout=10.0
    )


def generate_GC_frames(
    x_bound: Tuple[float, float], z_bound: Tuple[float, float], density=10
) -> List[RigidTransform]:

    xl = x_bound[1] / (density / 2.0)

    x_range = np.linspace(x_bound[0], x_bound[1], num=density)
    z_range = np.linspace(z_bound[0], z_bound[1], num=density)
    n = int((0.22 / 0.3) * density - 1)
    z_range[n] = 0.22

    X_GC_all = []
    for i in range(density):
        for j in range(density):
            X_GC_all.append(RigidTransform([x_range[i], 0, z_range[j]]))
    return X_GC_all


def score(b: state.Belief, CF_d=contact_defs.ground_align) -> int:
    return sum([int(p.satisfies_contact(CF_d)) for p in b.particles])


def simulate_GC_frames(
    b0: state.Belief,
    x_bound: Tuple[float, float],
    z_bound: Tuple[float, float],
    density: int,
) -> ExperimentResult:

    X_WG_nom = utils.xyz_rpy_deg([0.5, 0.0, 0.22], [180, 0, 0])
    K_nom = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 600.0])
    u_nom = components.CompliantMotion(RigidTransform(), X_WG_nom, K_nom)
    X_GC_all = generate_GC_frames(x_bound, z_bound, density=density)
    motions_all = [map_motion_to_GC_frame(u_nom, X_GC) for X_GC in X_GC_all]
    scores = []
    for motion in tqdm(motions_all):
        posterior = dynamics.f_bel(b0, motion)
        scores.append(
            (
                motion.X_GC.translation()[0],
                motion.X_GC.translation()[2],
                score(posterior),
            )
        )
    return scores


def visualize_experiment_result(
    result: ExperimentResult,
    x_bound: Tuple[float, float],
    z_bound: Tuple[float, float],
    density: int,
):
    block_width = 100
    im_size = (density + 1) * block_width
    im = np.zeros((im_size, im_size))

    for (x, z, score) in result:
        x_idx = ((x - x_bound[0]) / (x_bound[1] - x_bound[0])) * density
        start_x = int(x_idx * block_width)
        z_idx = ((z - z_bound[0]) / (z_bound[1] - z_bound[0])) * density
        start_z = int(z_idx * block_width)
        px_val = float(score / num_particles) * 255
        print(f"real: {x=}, {z=}")
        print(
            f"draw: ({start_x}, {start_x + block_width}), ({start_z}, {start_z + block_width}) {px_val}"
        )
        im[start_x : start_x + block_width, start_z : start_z + block_width].fill(
            px_val
        )

    im = im.T  # whoops
    plt.scatter(y=[0], x=[int(im_size / 2)], c="r", s=60)  # gripper
    block_z = (0.23 / 0.3) * im_size
    plt.scatter(y=[int(block_z)], x=[int(im_size / 2)], c="w", s=60)
    plt.imshow(im)
    # plt.show()
    plt.savefig("rcc.png")


def GC_experiment():
    b0 = b()
    x_bound = [-0.15, 0.15]
    z_bound = [-0.00, 0.30]
    density = 20
    simulation_output = simulate_GC_frames(b0, x_bound, z_bound, density)
    visualize_experiment_result(simulation_output, x_bound, z_bound, density)


def opt_rcc():
    b0 = b()
    X_GC_nom = RigidTransform([0.0, 0.0, 0.22])
    X_WC_nom = utils.xyz_rpy_deg([0.5, 0.0, 0.0], [180, 0, 0])
    K_nom = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 600.0])
    u_nom = components.CompliantMotion(X_GC_nom, X_WC_nom, K_nom, timeout=10.0)
    bnext = dynamics.f_bel(b0, u_nom)
    for p in bnext.particles:
        print(f"{p.contacts=}")
    print(f"{bnext.satisfies_contact(contact_defs.ground_align)=}")
    # p_out = dynamics.simulate(b0.particles[3], u_nom, vis=True)
    # print(f"{p_out.contacts=}")


@dataclass(frozen=True)
class Task:
    p: state.Particle
    u: components.CompliantMotion
    CF_d: components.ContactState


FAILED = (500, 500)


def process_task(t: Task) -> Tuple[float, float]:
    p_out = dynamics.simulate(t.p, t.u, vis=False)
    if p_out.satisfies_contact(t.CF_d):
        p_GC = t.u.X_GC.translation()
        return (p_GC[0], p_GC[2])
    else:
        return FAILED


def check_RCCs_parallel():
    X_WG_nom = utils.xyz_rpy_deg([0.5, 0.0, 0.22], [180, 0, 0])
    K_nom = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 600.0])
    u_nom = components.CompliantMotion(RigidTransform(), X_WG_nom, K_nom)
    p = multiprocessing.Pool(
        multiprocessing.cpu_count(),
        initializer=signal.signal,
        initargs=(signal.SIGINT, signal.SIG_IGN),
    )
    tasks = []
    particles = b().particles
    d = 11
    offset = 0.3 / (d - 1)
    xbins = np.linspace(-0.075, 0.075 + offset, num=d + 1)
    zbins = np.linspace(0.0, 0.3 + offset, num=d + 1)
    X_GC_all = generate_GC_frames([-0.075, 0.075], [0.0, 0.3], density=d)
    motions_all = [map_motion_to_GC_frame(u_nom, X_GC) for X_GC in X_GC_all]
    for motion in motions_all:
        for particle in particles:
            tasks.append(Task(particle, motion, contact_defs.ground_align))
    r = list(tqdm(p.imap_unordered(process_task, tasks), total=len(tasks)))
    xs = []
    ys = []
    for pt in r:
        if pt == FAILED:
            continue
        xs.append(pt[0])
        ys.append(pt[1])
    h = plt.hist2d(xs, ys, bins=[xbins, zbins])
    print(f"{np.max(h[0])=}")
    plt.colorbar(h[3])
    plt.savefig("rcc2.png")


if __name__ == "__main__":
    check_RCCs_parallel()
