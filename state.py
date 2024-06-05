from __future__ import annotations

import random
from multiprocessing import Process, Queue
from typing import Dict, List

import numpy as np
import numpy.linalg as la
from pydrake.all import RigidTransform, System

import components
from simulation import plant_builder

random.seed(0)


class Particle:
    def __init__(
        self,
        q_r: np.ndarray,
        X_GM: RigidTransform,
        X_WO: RigidTransform,
        env_geom: str,
        manip_geom: str,
        mu: float = 0.0,
    ):
        self.q_r = q_r
        self.X_GM = X_GM
        self.X_WO = X_WO
        self.env_geom = env_geom
        self.manip_geom = manip_geom
        self.mu = mu
        self._contacts = None
        self._sdf = None
        self._constraints = None
        self._X_WG = None
        self._manip_poly = None
        self.trajectory = []
        self._sim_id = None
        self.cspace_repr = None
        self.noisy = False

    def make_plant(
        self,
        vis: bool = False,
        collision: bool = False,
        meshcat_instance=None,
    ) -> System:
        return plant_builder.make_plant(
            self.q_r,
            self.X_GM,
            self.X_WO,
            self.env_geom,
            self.manip_geom,
            vis=vis,
            collision_check=collision,
            mu=self.mu,
            meshcat_instance=meshcat_instance,
        )

    def _update_contact_data(self):
        diagram, _ = self.make_plant(collision=True)
        diagram_context = diagram.CreateDefaultContext()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyContextFromRoot(diagram_context)
        plant.SetPositions(
            plant_context, plant.GetModelInstanceByName("panda"), self.q_r
        )
        geom_monitor = diagram.GetSubsystemByName("geom_monitor")
        geom_monitor.ForcedPublish(geom_monitor.GetMyContextFromRoot(diagram_context))
        self._contacts = geom_monitor.contacts
        self._sdf = geom_monitor.sdf
        self._constraints = geom_monitor.constraints
        self._manip_poly = geom_monitor.manip_poly

    @property
    def contacts(self) -> components.ContactState:
        if self._contacts is None:
            self._update_contact_data()
        return self._contacts

    @property
    def sdf(self) -> Dict[components.Contact, float]:
        if self._sdf is None:
            self._update_contact_data()
        return self._sdf

    @property
    def constraints(self):
        if self._constraints is None:
            self._update_contact_data()
        return self._constraints

    def compute_X_WG(self, q):
        diagram, _ = self.make_plant()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
        plant.SetPositions(
            plant_context, plant.GetModelInstanceByName("panda"), self.q_r
        )
        _X_WG_cand = plant.CalcRelativeTransform(
            plant_context,
            plant.world_frame(),
            plant.GetBodyByName("panda_hand").body_frame(),
        )
        q.put(_X_WG_cand)

    @property
    def X_WG(self):
        if self._X_WG is None:
            q = Queue()
            p = Process(target=self.compute_X_WG, args=(q,))
            p.start()
            p.join()
            self._X_WG = q.get()
        return self._X_WG

    @property
    def X_WM(self) -> RigidTransform:
        return self.X_WG.multiply(self.X_GM)

    @property
    def X_OM(self) -> RigidTransform:
        return self.X_WO.InvertAndCompose(self.X_WM)

    def deepcopy(self) -> Particle:
        new_p = Particle(
            self.q_r.copy(),
            self.X_GM,
            self.X_WO,
            self.env_geom,
            self.manip_geom,
            mu=self.mu,
        )
        new_p._constraints = self._constraints
        new_p._manip_poly = self._manip_poly
        new_p._sim_id = self._sim_id
        new_p.noisy = self.noisy
        return new_p

    def satisfies_contact(self, CF_d: components.ContactState, epsilon=0.001) -> bool:
        sdf = self.sdf
        for contact in CF_d:
            if sdf.get(contact, 0.1) > epsilon:
                return False
        return True

    def epsilon_contacts(self, epsilon: float = 0.001) -> components.ContactState:
        contacts = []
        for cf, dist in self.sdf.items():
            if dist < epsilon:
                contacts.append(cf)
        return frozenset(contacts)


class Belief:
    def __init__(self, particles: List[Particle]):
        self.particles = particles

    def sample(self) -> Particle:
        return random.choice(self.particles)

    def _contact_sat_dbg(self, CF_d: components.ContactState, epsilon=0.001):
        satisfies = True
        for p in self.particles:
            if not p.satisfies_contact(CF_d, epsilon=epsilon):
                print("csat failed")
                satisfies = False
            else:
                print("csat succ")
        return satisfies

    def satisfies_contact(self, CF_d: components.ContactState, epsilon=0.001) -> bool:
        for p in self.particles:
            if not p.satisfies_contact(CF_d, epsilon=epsilon):
                return False
        return True

    def partial_sat_score(self, CF_d) -> float:
        score = 0.0
        for p in self.particles:
            for contact in CF_d:
                if p.satisfies_contact(set((contact,))):
                    score += 1.0 / len(CF_d)
        return score + 0.001 * self.iou()

    def contact_state(self, epsilon=0.001) -> components.ContactState:
        assert len(self.particles) > 0
        cs = self.particles[0].epsilon_contacts(epsilon)
        for i in range(1, len(self.particles)):
            cs = cs.intersection(self.particles[i].epsilon_contacts(epsilon))
        return frozenset(cs)

    def iou(self, epsilon=0.001) -> float:
        intersection = self.particles[0].epsilon_contacts(epsilon)
        union = self.particles[0].epsilon_contacts(epsilon)
        for i in range(1, len(self.particles)):
            intersection = intersection.intersection(
                self.particles[i].epsilon_contacts(epsilon)
            )
            union = union.union(self.particles[i].epsilon_contacts(epsilon))
        if len(union) == 0:
            return 0.0
        return len(intersection) / len(union)

    def mean(self) -> Particle:
        avg_xyz = (1.0 / len(self.particles)) * sum(
            [p.X_WG.translation() for p in self.particles]
        )
        best_particle = None
        smallest_diff = float("inf")
        for p in self.particles:
            diff = np.linalg.norm(p.X_WG.translation() - avg_xyz)
            if diff < smallest_diff:
                smallest_diff = diff
                best_particle = p
        assert best_particle is not None
        del avg_xyz
        return best_particle

    def mean_translation(self) -> np.ndarray:
        avg_xyz = (1.0 / len(self.particles)) * sum(
            [p.X_WG.translation() for p in self.particles]
        )
        return avg_xyz

    def direction(self) -> np.ndarray:
        qs = np.array([p.X_WM.translation() for p in self.particles])
        qs_normalized = qs - np.mean(qs)
        cov = np.cov(qs_normalized, rowvar=True)
        evals, evecs = la.eig(cov)
        idx = np.argsort(evals)
        return evecs[idx[0]]
