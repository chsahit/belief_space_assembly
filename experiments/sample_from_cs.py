import numpy as np
from pydrake.all import RigidTransform

import components
import contact_defs
import dynamics
import puzzle_contact_defs
import visualize
from experiments import init_particle
from simulation import generate_contact_set, ik_solver


def sample_from_cs(CF_d: components.ContactState, n: int = 1):
    p = init_particle.init_peg()
    X_WGs = generate_contact_set.project_manipuland_to_contacts(p, CF_d, num_samples=n)
    U = []
    for X_WG in X_WGs:
        t2 = np.copy(X_WG.translation())
        # t2[2] -= 0.0001
        X_WGd = RigidTransform(X_WG.rotation(), t2)
        u = components.CompliantMotion(RigidTransform(), X_WGd, components.stiff)
        print(f"{X_WG.translation()=}")
        u = ik_solver.update_motion_qd(u)
        U.append(u)
    if n == 1:
        print("visualizing setpoint")
        visualize.visualize_targets(p, X_WGs)
        print("visualizing control")
        p_next = dynamics.simulate(p, u, vis=True)
        print(f"{p_next.satisfies_contact(CF_d)=}")
        print(f"{p_next.epsilon_contacts()=}")
    else:
        posteriors = dynamics.f_cspace(p, U)
        score = 0
        for post in posteriors:
            if post.satisfies_contact(CF_d):
                score += 1
        print(f"{score=}")


if __name__ == "__main__":
    sample_from_cs(contact_defs.chamfer_init, n=32)
    input()
