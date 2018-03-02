import tensorflow as tf
import numpy as np
from data.model_files.fcrn_DepthPrediction import models
from PIL import Image

@DeprecationWarning
class DepthDetector:
    def __init__(self, ckpt_file_path, width, height):
        # pending for implementation
        self.width = width
        self.height = height
        self.channels = 3
        self.batch_size = 1
        self.input_node = tf.placeholder(tf.float32, shape=(None, height, width, self.channels))

        print('creating depth detection graph')
        self.output_tensor = models.ResNet50UpProj({'data': self.input_node}, self.batch_size, 1, False).get_output()

        print('loading depth detection parameters')
        self.session = tf.Session()
        tf.train.Saver().restore(self.session, ckpt_file_path)

    def detect_depth_for_image(self, image):
        image = Image.fromarray(image)
        image = image.resize((self.height, self.width), Image.ANTIALIAS)
        image = np.array(image)
        image = np.expand_dims(image, axis=0)

        print(image.shape)

        depth = self.session.run(self.output_tensor, feed_dict={self.input_node: image})

        depth = depth.reshape(depth.shape[1:3])
        depth = Image.fromarray(depth)
        depth = depth.resize((self.height, self.width), Image.ANTIALIAS)

        depth = np.asarray(depth)

        return depth
