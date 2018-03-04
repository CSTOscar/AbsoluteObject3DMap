@DeprecationWarning
class ObjectProjection:
    id_seed = 0

    def __init__(self, frame_id, clazz, confidence, position, err_cov):
        self.id = ObjectProjection.get_unique_id()
        self.frame_id = frame_id
        self.clazz = clazz
        self.confidence = confidence
        self.position = position
        self.err_cov = err_cov

    @classmethod
    def get_unique_id(cls):
        cls.id_seed += 1
        return cls.id_seed

    def __str__(self):
        return 'id: ' + str(self.id) \
               + '\nframe_id: ' \
               + str(self.frame_id) \
               + '\nclass: ' \
               + str(self.clazz) \
               + '\nconfidence: \n' \
               + str(self.confidence) \
               + '\nposition : \n' \
               + str(self.position) \
               + '\nerr_cov: \n' \
               + str(self.err_cov)

    def distance(self, obj):
        return
