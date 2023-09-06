from pydrake.all import AbstractValue, EventStatus, ImageRgba8U, LeafSystem


class ImageLogger(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.rgb_image_output_port = self.DeclareAbstractInputPort(
            "rbg_in", AbstractValue.Make(ImageRgba8U())
        )
        self.last_image = None
        self.DeclarePerStepPublishEvent(self.update_image)

    def update_image(self, context):
        self.last_image = 1
        return EventStatus.Succeeded()
