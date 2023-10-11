import sys

import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import state
import utils
import visualize
from planning import refine_motion
from simulation import ik_solver

mostly_soft = np.array([10.0, 10.0, 10.0, 100.0, 100.0, 600.0])


def init(X_GM_x: float = 0.0, X_GM_p: float = 0.0) -> RigidTransform:
    X_WG_0 = utils.xyz_rpy_deg([0.5, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([X_GM_x, 0.0, 0.155], [0, X_GM_p, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/big_chamfered_hole.sdf", "assets/peg.urdf", mu=0.6
    )
    return p_0


def init_motion(K: np.ndarray = components.stiff) -> components.CompliantMotion:
    X_WC_d = utils.xyz_rpy_deg([0.5, 0.0, 0.0], [180, 0, 0])
    X_GC = utils.xyz_rpy_deg([0.0, 0.0, 0.23], [0, 0, 0])
    return components.CompliantMotion(X_GC, X_WC_d, K)


def test_no_noise():
    p0 = init()
    X_WG_d = utils.xyz_rpy_deg([0.5, 0.0, 0.2], [180, 0, 0])
    # u0 = components.CompliantMotion(RigidTransform(), X_WG_d, components.stiff)
    u0 = init_motion(components.soft)
    p1 = dynamics.simulate(p0, u0, vis=True)


def test_with_noise():
    p0 = init(X_GM_x=0.01, X_GM_p=-10.0)
    u0 = init_motion(K=components.soft)
    p1 = dynamics.simulate(p0, u0, vis=True)


def test_with_tp():
    p0 = init(X_GM_x=0.01, X_GM_p=-10.0)
    # motion 0
    X_GC = utils.xyz_rpy_deg([-0.03, 0, 0.23], [0, 0, 0])
    X_WC_d = utils.xyz_rpy_deg([0.47, 0.0, 0.1], [180, 0, 0])
    u0 = components.CompliantMotion(X_GC, X_WC_d, components.soft)
    # motion 1
    X_WC_d = utils.xyz_rpy_deg([0.47, 0.0, 0.0], [180, 0, 0])
    u1 = components.CompliantMotion(X_GC, X_WC_d, mostly_soft)
    # simulate
    p1 = dynamics.simulate(p0, u0, vis=True)
    p2 = dynamics.simulate(p1, u1, vis=True)


# test_searches(contact_defs.lc_align)
def test_single_refine(CF_d: components.ContactState):
    p_a = init()
    p_b = init(X_GM_x=0.01, X_GM_p=-10.0)
    b = state.Belief([p_a, p_b])
    u = refine_motion.refine(b, CF_d)
    if u is not None:
        b_result = dynamics.f_bel(b, u)
        print(b_result.satisfies_contact(CF_d))
    else:
        print("search failed")


def please_assemble():  # ...please?
    p_a = init()
    p_b = init(X_GM_x=0.01, X_GM_p=-10.0)
    b = state.Belief([p_a, p_b])
    # modes = [contact_defs.fc_align, contact_defs.f_align]
    """
    modes = [
        contact_defs.f_full_chamfer_touch,
        contact_defs.corner_align_2,
        contact_defs.ground_align,
    ]
    """
    modes = [
        contact_defs.b_full_chamfer_touch,
        contact_defs.bf_only_align,
        contact_defs.ff_only_align,
        contact_defs.ground_align,
    ]

    for mode in modes:
        u = refine_motion.refine(b, mode)
        if u is not None:
            b_next = dynamics.f_bel(b, u)
            for p in b.particles:
                dynamics.simulate(p, u, vis=True)
            b = b_next
        else:
            print("search failed")
            sys.exit()
    input()


def ik_debug():
    p = init()
    X_WG = ik_solver.project_manipuland_to_contacts(p, contact_defs.corner_align_2)
    q_r_solved = ik_solver.gripper_to_joint_states(X_WG)
    p_solved = p.deepcopy()
    p_solved.q_r = q_r_solved
    im = visualize.generate_particle_picture(p_solved)
    im.save("ff_align.jpg")
    visualize.show_particle(p_solved)


if __name__ == "__main__":
    please_assemble()
