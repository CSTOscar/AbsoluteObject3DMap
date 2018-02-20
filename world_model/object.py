class Object:
    id_seed = 0

    def __init__(self, frame_id, clazz, confidence, position, err_cov):
        self.id = Object.get_unique_id()
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
               + '\n frame_id: ' \
               + str(self.frame_id) \
               + '\n class: ' \
               + str(self.clazz) \
               + '\n confidence: \n' \
               + str(self.confidence) \
               + '\n position : \n' \
               + str(self.position) \
               + '\n err_cov: \n' \
               + str(self.err_cov)
