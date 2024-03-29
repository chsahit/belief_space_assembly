from __future__ import annotations

import random
from itertools import product
from typing import Dict, List

import numpy as np
import numpy.linalg as la
from pydrake.all import HPolyhedron, RigidTransform, Simulator, System, VPolytope

import components
from simulation import plant_builder


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
        self._J = None

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
        self._J = geom_monitor.J
        self._constraints = geom_monitor.constraints
        self._manip_poly = geom_monitor.manip_poly

    @property
    def J(self) -> np.ndarray:
        if self._J is None:
            self._update_contact_data()
        return self._J

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

    @property
    def X_WG(self):
        if self._X_WG is None:
            diagram, _ = self.make_plant()
            plant = diagram.GetSubsystemByName("plant")
            plant_context = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
            plant.SetPositions(
                plant_context, plant.GetModelInstanceByName("panda"), self.q_r
            )
            self._X_WG = plant.CalcRelativeTransform(
                plant_context,
                plant.world_frame(),
                plant.GetBodyByName("panda_hand").body_frame(),
            )
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
        return score + 0.001 * len(self.contact_state())

    def contact_state(self, epsilon=0.001) -> components.ContactState:
        assert len(self.particles) > 0
        cs = self.particles[0].epsilon_contacts(epsilon)
        for i in range(1, len(self.particles)):
            cs = cs.intersection(self.particles[i].epsilon_contacts(epsilon))
        filtered_cs = []
        for c in cs:
            if "Box" not in str(c) or True:
                filtered_cs.append(c)
        filtered_cs = frozenset(filtered_cs)
        return filtered_cs

    def score(
        self,
        CF_d: components.ContactState,
        epsilon: float = 0.001,
        delta: float = 0.001,
    ) -> float:
        assert len(self.particles) > 0
        num_sat = [
            int(p.satisfies_contact(CF_d, epsilon=epsilon)) for p in self.particles
        ]
        ratio_satisfiying = float(sum(num_sat)) / float(len(self.particles))
        num_contacts = len(self.contact_state())
        return ratio_satisfiying + (delta * num_contacts)

    @staticmethod
    def make_particles(
        grasps: List[components.Grasp],
        O_poses: List[components.ObjectPose],
        nominal: Particle,
    ) -> Belief:
        assert len(grasps) == len(O_poses)
        particles = []
        for i in range(len(grasps)):
            X_GM = grasps[i].get_tf()
            X_WO = O_poses[i].get_tf()
            particles.append(
                Particle(
                    nominal.q_r,
                    X_GM,
                    X_WO,
                    nominal.env_geom,
                    nominal.manip_geom,
                    mu=nominal.mu,
                )
            )
        return Belief(particles)

    def mean(self) -> state.Particle:
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
        return best_particle

    def direction(self) -> np.ndarray:
        qs = np.array([p.X_WM.translation() for p in self.particles])
        qs_normalized = qs - np.mean(qs)
        cov = np.cov(qs, rowvar=True)
        evals, evecs = la.eig(cov)
        idx = np.argsort(evals)
        return evecs[idx[0]]
