from typing import List

import numpy as np
from PIL import Image
from pydrake.all import Simulator

import components
from belief import belief_state, dynamics
from simulation import plant_builder


def render_motion_set(
    p_nominal: belief_state.Particle, U: List[components.CompliantMotion]
):
    p = p_nominal.deepcopy()
    p.env_geom = "assets/empty_world.sdf"
    U_stiff = [components.CompliantMotion(u.X_GC, u.X_WCd, components.stiff) for u in U]
    P_out = f_cspace(p, U_stiff)
    images = [generate_particle_picture(p_i) for p_i in P_out]


def generate_particle_picture(p: belief_state.Particle, name="test.jpg") -> Image:
    diagram = plant_builder.make_plant_with_cameras(
        p.q_r, p.X_GM, p.X_WO, p.env_geom, p.manip_geom
    )
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.AdvanceTo(0.1)
    logger = diagram.GetSubsystemByName("camera_logger")
    im = Image.fromarray(logger.last_image)
    return im
