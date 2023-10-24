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
        u, succ = refine_motion.refine(b_curr, mode)
        if u is None:
            print("search failed. aborting")
            break
        soln.append(u)
        b_curr = succ

    return soln
