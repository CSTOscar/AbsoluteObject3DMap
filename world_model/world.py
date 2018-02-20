import frame.frame as fm


class World:
    def __init__(self):
        self.objects_projection = []
        self.objects = []

    def add_objects_from_frame(self, frame: fm, confidence):
        self.objects_projection.extend(frame.get_objects_with_confidence_more_than(confidence))

    def unify_objects_projection_get_object(self):
        return []
