import time

import numpy as np
from pydrake.all import RigidTransform

import components
import dynamics
import state
import utils
import visualize
from experiments import init_particle
from simulation import ik_solver


def init(X_GM_x: float = 0.0):
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.155], [180, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/big_chamfered_hole.sdf", "assets/peg.urdf"
    )
    return p_0


def test_simulate():
    p_0 = init_particle.init_peg()  # init()
    # visualize.project_to_planar(p_0)
    t0 = time.time()
    X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.30], [180, 0, 0])
    u_0 = components.CompliantMotion(RigidTransform(), X_WG_d, components.stiff)
    # print(utils.RigidTfToVec(p_0.X_WG))
    dynamics.simulate(p_0, u_0, vis=True)
    tT = time.time()
    # print(f"{p_1.contacts=}")
    print(f"sim time={tT - t0}")
    input()


def test_parallel_sim():
    p0 = init(X_GM_x=-0.002)
    p1 = init(X_GM_x=0.002)
    b = state.Belief([p0, p1])
    U = [
        components.CompliantMotion(
            RigidTransform(),
            utils.xyz_rpy_deg([0.5, 0.0, 0.22], [180, 0, 0]),
            components.stiff,
        )
    ]
    p_out = dynamics.simulate(b.particles[0], U[0], vis=True)
    print(f"{p_out.X_WG=}")
    visualize.play_motions_on_belief(b, U)
    input()


def test_motion_set():
    p0 = init()
    zs = np.linspace(0.2, 0.3, 5)
    rts = [utils.xyz_rpy_deg([0.5, 0.0, z], [180, 0, 0]) for z in zs]
    U = [
        components.CompliantMotion(RigidTransform(), rt, components.stiff) for rt in rts
    ]
    result_set = dynamics.f_cspace(p0, U)
    for i, pT in enumerate(result_set):
        print(f"particle {i}: {pT.contacts}")


def test_vis():
    p0 = init()
    p0.q_r = np.array(
        [
            0.19045906,
            1.53978349,
            -0.14104434,
            -0.0698,
            -0.20832431,
            0.66944977,
            0.3623883,
            0.0,
            0.0,
        ]
    )
    p0.env_geom = "assets/empty_world.sdf"
    im = visualize.generate_particle_picture(p0)
    im.save("test.jpg")


def funny_rcc():
    p0 = init()
    X_GC = RigidTransform([-0.05, 0.0, 0.0])
    X_WCd = utils.xyz_rpy_deg([0.45, 0.0, 0.22], [180, 0, 0])
    K_nom = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 600.0])
    u_nom = components.CompliantMotion(X_GC, X_WCd, K_nom)
    dynamics.simulate(p0, u_nom, vis=True)


if __name__ == "__main__":
    test_simulate()
    # funny_rcc()
    # test_vis()
