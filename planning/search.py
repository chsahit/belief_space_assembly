from typing import List

import components
import dynamics
import state
from planning import refine_motion


def refine_schedule(
    b0: state.Belief,
    g: components.ContactState,
    schedule: List[components.ContactState],
) -> List[components.CompliantMotion]:

    soln = []
    b_curr = b0
    for mode in schedule:
        if b_curr.satisfies_contact(mode):
            print(f"contact {mode} already satisfied")
            continue
        print(f"targeting mode: ", mode)
        u, succ = refine_motion.refine(b_curr, mode)
        for prior_p in b_curr.particles:
            print("simulating from refinement motion: ")
            dynamics.simulate(prior_p, u, vis=True)
            print("-----")
        if u is None:
            print("search failed. aborting")
            break
        soln.append(u)
        b_curr = succ

    return soln
