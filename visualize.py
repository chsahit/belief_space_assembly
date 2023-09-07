from typing import List

import numpy as np
from PIL import Image
from pydrake.all import Simulator

import components
import dynamics
import state
from simulation import plant_builder


def _merge_images(images) -> Image:
    base = np.full((1080, 1080, 3), (204, 230, 255), dtype=np.uint8)
    for im in images:
        i = np.array(im)
        bg_color = i[0, 0]
        bg_full = np.full((1080, 1080, 3), bg_color, dtype=np.uint8)
        bg_diff = np.abs(i - bg_full)
        fg_mask = (bg_diff > [1, 1, 1]).any(-1)
        foreground = (i != bg_color).all(-1)
        base[fg_mask] = i[fg_mask]
    return Image.fromarray(base)


def render_motion_set(p_nominal: state.Particle, U: List[components.CompliantMotion]):
    p = p_nominal.deepcopy()
    p.env_geom = "assets/empty_world.sdf"
    U_stiff = [components.CompliantMotion(u.X_GC, u.X_WCd, components.stiff) for u in U]
    P_out = dynamics.f_cspace(p, U_stiff)
    images = [generate_particle_picture(p_i) for p_i in P_out]

    composite = _merge_images(images)
    composite.save("composite.png")


def generate_particle_picture(p: state.Particle, name="test.jpg") -> Image:
    diagram = plant_builder.make_plant_with_cameras(
        p.q_r, p.X_GM, p.X_WO, p.env_geom, p.manip_geom
    )
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.AdvanceTo(0.1)
    logger = diagram.GetSubsystemByName("camera_logger")
    im = Image.fromarray(logger.last_image)
    return im
