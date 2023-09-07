import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import state
import utils
import visualize
from planning import motion_sets, refine_motion
from simulation import ik_solver, plant_builder


def init_state() -> state.Particle:
    X_WG_0 = utils.xyz_rpy_deg([0.50, 0.0, 0.36], [180, 0, 0])
    X_GM = utils.xyz_rpy_deg([0.0, 0.0, 0.155], [0, 0, 0])
    X_WO = utils.xyz_rpy_deg([0.5, 0, 0.075], [0, 0, 0])
    q_r_0 = ik_solver.gripper_to_joint_states(X_WG_0)
    p_0 = state.Particle(
        q_r_0, X_GM, X_WO, "assets/chamfered_hole.sdf", "assets/peg.urdf"
    )
    return p_0


def init_motion() -> components.CompliantMotion:
    X_WC_d = utils.xyz_rpy_deg([0.5, 0.0, -0.01], [180, 0, 0])
    X_GC = utils.xyz_rpy_deg([0, 0, 0.23], [0, 0, 0])
    return components.CompliantMotion(X_GC, X_WC_d, components.stiff)


def test_goal_sat():
    p0 = init_state()
    u0 = init_motion()
    p_out = dynamics.simulate(p0, u0, vis=True)
    print(p_out.satisfies_contact(contact_defs.ground_align))


def test_compute_compliance_frame():
    p_nom = init_state()
    cspheres = plant_builder.generate_collision_spheres()
    X_GC = refine_motion.compute_compliance_frame(
        p_nom.X_GM, contact_defs.ground_align, cspheres
    )
    print(X_GC)


def test_grow_motion_set():
    u_nom = init_motion()
    X_GC = utils.xyz_rpy_deg([0, 0, 0.23], [0, 0, 0])
    p = init_state()
    U = motion_sets.grow_motion_set(
        X_GC, components.stiff, contact_defs.ground_align, p, density=2
    )
    print(f"{len(U)=}")
    return U


def test_view_motion_set():
    p_nom = init_state()
    U = test_grow_motion_set()
    visualize.render_motion_set(p_nom, U)


def test_intersect_motion_set():
    grasps = [
        components.Grasp(x=-0.01, z=0.155, pitch=0),
        components.Grasp(x=0.01, z=0.155, pitch=0),
    ]
    bin_poses = [components.ObjectPose(x=0.5, y=0, yaw=0)] * len(grasps)
    X_GC = utils.xyz_rpy_deg([0, 0, 0.23], [0, 0, 0])
    p_nom = init_state()
    b = state.Belief.make_particles(grasps, bin_poses, p_nom)
    u_res = motion_sets.intersect_motion_sets(
        X_GC, components.stiff, b, contact_defs.ground_align
    )
    b_next = dynamics.f_bel(b, u_res)
    print([p.contacts for p in b_next.particles])


if __name__ == "__main__":
    test_intersect_motion_set()
