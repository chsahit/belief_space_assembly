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
        u = refine_motion.refine(b_curr, mode)
        soln.append(u)
        b_curr = dynamics.f_bel(b_curr, u)

    return soln
