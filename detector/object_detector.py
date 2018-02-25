import numpy as np
import tensorflow as tf
from research.object_detection.utils import label_map_util


# A persist object which provide a detection info

class ObjectorDetector:
    def __init__(self, ckpt_file_path, label_file_path, num_classes):
        print('Loading tensorflow graph')
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(ckpt_file_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        print('Loading label file')

        self.label_map = label_map_util.load_labelmap(label_file_path)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=num_classes,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def detect_object(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                image = np.expand_dims(image, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image})
            return ObjectorDetector.create_image_detection_record(boxes[0], scores[0], classes[0], num[0])

    @staticmethod
    def create_image_detection_record(boxes, scores, classes, num):
        record = []
        for i in range(int(num)):
            record.append({'box': boxes[i], 'score': scores[i], 'class': classes[i]})
        sorted(record, key=lambda e: e['score'], reverse=True)
        return record
