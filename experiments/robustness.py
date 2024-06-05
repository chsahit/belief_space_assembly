import logging
import multiprocessing
import multiprocessing.pool

from tqdm import tqdm

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
planners = {"cobs": cobs.cobs, "b_est": ao_b_est.b_est, "no_k": cobs.no_k}
logging.basicConfig(level=logging.WARNING)


# https://stackoverflow.com/questions/52948447/error-group-argument-must-be-none-for-now-in-multiprocessing-pool
class NonDaemonPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)

        class NonDaemonProcess(proc.__class__):
            """Monkey-patch process to ensure it is never daemonized"""

            @property
            def daemon(self):
                return False

            @daemon.setter
            def daemon(self, val):
                pass

        proc.__class__ = NonDaemonProcess
        return proc


def make_b0(peg_size: str, noisy: bool = False) -> state.Belief:
    peg_size = "normal"
    urdf = peg_urdfs[peg_size]
    p0 = init_particle.init_peg(X_GM_x=-0.01, peg_urdf=urdf)
    p0.noisy = noisy
    p1 = init_particle.init_peg(pitch=0, peg_urdf=urdf)
    p1.noisy = noisy
    p2 = init_particle.init_peg(X_GM_x=0.01, peg_urdf=urdf)
    p2.noisy = noisy
    return state.Belief([p0, p1, p2])


def run_test(planner_name: str, goal: components.ContactState):
    b0 = make_b0("normal")
    plan_result = planners[planner_name](b0, goal)
    b0_test = make_b0("normal", noisy=True)
    bT_test = dynamics.sequential_f_bel(b0_test, plan_result.traj)[-1]
    if bT_test.satisfies_contact(goal):
        print("return 1.0")
        return 1.0
    print("return 0.0")
    return 0.0


def parallel_sweep(planner_name: str, trials: int = 25):
    p = NonDaemonPool(8)
    arg_list = [(planner_name, contact_defs.peg_goal) for i in range(trials)]
    scores = p.starmap(run_test, arg_list)
    p.close()
    p.join()
    sr = float(sum(scores)) / float(trials)
    print(f"{planner_name}, {sr}")


def sweep_geoms(planner_name: str):
    results = []
    for geom_size in ["big"]:
        trials = 25
        num_succs = 0.0
        for trial in tqdm(range(trials)):
            b0 = make_b0("normal")
            goal = contact_defs.peg_goal
            planner = planners[planner_name]
            plan_result = planner(b0, goal)
            assert plan_result.traj is not None
            b0_test = make_b0(geom_size, noisy=True)
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
    parallel_sweep("b_est")
