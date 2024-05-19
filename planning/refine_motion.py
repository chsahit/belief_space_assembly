from typing import List

import components
import dynamics
import state
from planning import randomized_search

sample_logs_rr = []


def randomized_refine(
    b: state.Belief,
    modes: List[components.ContactState],
    search_compliance: bool = True,
    do_gp: bool = True,
    max_attempts: int = 3,
) -> List[components.CompliantMotion]:
    global sample_logs_rr
    last_refined = None
    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"{attempt=}")
        curr = b
        traj = []
        for i, mode in enumerate(modes):
            if i == 0:
                last_refined = (None, mode)
            else:
                last_refined = (modes[i - 1], modes[i])
            print(f"{mode=}")
            m_best_score = 0.0
            while m_best_score < len(b.particles):
                u_star = randomized_search.refine_b(
                    curr, mode, search_compliance, do_gp
                )
                if u_star is None:
                    break
                curr_tenative = dynamics.f_bel(curr, u_star)
                curr_best_score = curr_tenative.partial_sat_score(mode)
                if curr_best_score <= m_best_score + 1e-6:
                    break
                m_best_score = curr_best_score
                curr = curr_tenative
                traj.append(u_star)
                sample_logs_rr = randomized_search.sample_logs
            if u_star is None:
                break
        if curr.satisfies_contact(modes[-1]):
            return components.PlanningResult(traj, 0, 0, 0, last_refined)
    sample_logs_rr = []
    randomized_search.sample_logs = []
    return components.PlanningResult(None, 0, 0, 0, last_refined)
