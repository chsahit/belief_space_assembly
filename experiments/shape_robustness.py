from typing import List

import components
import contact_defs
import dynamics
import state
from experiments import init_particle
from planning import ao_b_est, cobs

peg_urdfs = {
    "small": "assets/small_peg.urdf",
    "normal": "assets/peg.urdf",
    "big": "assets/big_peg.urdf",
}
planners = {"cobs": cobs.cobs, "b_est": ao_b_est.b_est}


def make_b0(peg_size: str) -> state.Belief:
    urdf = peg_urdfs[peg_size]
    p0 = init_particle.init_peg(pitch=-2, peg_urdf=urdf)
    p1 = init_particle.init_peg(pitch=0, peg_urdf=urdf)
    p2 = init_particle.init_peg(pitch=2, peg_urdf=urdf)
    return state.Belief([p0, p1, p2])


def sweep_geoms(planner_name: str):
    for geom_size in ["small", "big"]:
        trials = 10
        num_succs = 0.0
        for trial in range(trials):
            b0 = make_b0("normal")
            goal = contact_defs.bottom_faces_2
            planner = planners[planner_name]
            plan_result = planner(b0, goal)
            assert plan_result.traj is not None
            b0_test = make_b0(geom_size)
            bT_test = dynamics.sequential_f_bel(b0_test, plan_result.traj)[-1]
            if bT_test.satisfies_contact(goal):
                num_succs += 1.0
        sr = num_succs / trials
        print(f"{planner_name=}, {geom_size=}, {sr=}")


if __name__ == "__main__":
    sweep_geoms("cobs")
