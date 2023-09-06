import numpy as np
from pydrake.all import AbstractValue, EventStatus, ImageRgba8U, LeafSystem


class ImageLogger(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.rgb_image_input_port = self.DeclareAbstractInputPort(
            "rbg_in", AbstractValue.Make(ImageRgba8U())
        )
        self.last_image = None
        self.DeclarePerStepPublishEvent(self.update_image)

    def update_image(self, context):
        image = self.rgb_image_input_port.Eval(context)
        contig_rgb_im = np.ascontiguousarray((image.data[:, :, :3]).astype(np.uint8))
        self.last_image = contig_rgb_im
        return EventStatus.Succeeded()
