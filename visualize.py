import numpy as np
from pydrake.all import Simulator

from belief import belief_state
from simulation import plant_builder


def save_particle_picture(p: belief_state.Particle):
    diagram = plant_builder.make_plant_with_cameras(
        p.q_r, p.X_GM, p.X_WO, p.env_geom, p.manip_geom
    )
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.AdvanceTo(0.1)
    logger = diagram.GetSubsystemByName("camera_logger")
    # white = 255 * np.ones_like(logger.last_image)
    # print(f"{logger.last_image=}")
    # print(f"{np.sum((white - logger.last_image).flatten())=}")
    from PIL import Image

    im = Image.fromarray(logger.last_image)
    im.save("test.png")
