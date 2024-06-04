import logging

from tqdm import tqdm

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
planners = {"cobs": cobs.cobs, "b_est": ao_b_est.b_est, "no_k": cobs.no_k}
logging.basicConfig(level=logging.WARNING)


def make_b0(peg_size: str) -> state.Belief:
    urdf = peg_urdfs[peg_size]
    p0 = init_particle.init_peg(X_GM_x=-0.01, peg_urdf=urdf)
    p1 = init_particle.init_peg(pitch=0, peg_urdf=urdf)
    p2 = init_particle.init_peg(X_GM_x=0.01, peg_urdf=urdf)
    return state.Belief([p0, p1, p2])


def sweep_geoms(planner_name: str):
    results = []
    for geom_size in ["big"]:
        trials = 25
        num_succs = 0.0
        for trial in tqdm(range(trials)):
            b0 = make_b0("normal")
            goal = contact_defs.bottom_faces_2
            planner = planners[planner_name]
            plan_result = planner(b0, goal)
            assert plan_result.traj is not None
            b0_test = make_b0(geom_size)
            bT_test = dynamics.sequential_f_bel(b0_test, plan_result.traj)[-1]
            if planner_name == "cobs" and geom_size == "big" and trial == 0:
                p_vis = b0_test.particles[0]
                for u in plan_result.traj:
                    p_vis = dynamics.simulate(p_vis, u, vis=True)
            if bT_test.satisfies_contact(goal):
                num_succs += 1.0
        sr = num_succs / trials
        results.append((planner_name, geom_size, sr))
    print(f"\n{results}")


if __name__ == "__main__":
    sweep_geoms("cobs")
    sweep_geoms("b_est")
