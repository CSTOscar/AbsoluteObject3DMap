import numpy as np
import os
import tensorflow as tf
from PIL import Image
from research.object_detection.utils import label_map_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

DATA_DIR = '../data'

LABEL_FILE_DIR = os.path.join(DATA_DIR, 'label_files')
MODEL_FILE_DIR = os.path.join(DATA_DIR, 'model_files')
IMAGE_DIR = os.path.join(DATA_DIR, 'temp_files/images')
RESULT_DIR = os.path.join(DATA_DIR, 'temp_files/results')
IMAGE_NAME_FORMAT = 'image{}.jpg'
DETECTION_RESULT_FILE_NAME = 'image_detection_record'


def detection(model_name, ckpt_file_name, label_file_name, max_num_classes, image_name_format, num_image):
    ckpt_file_path = os.path.join(MODEL_FILE_DIR, model_name, ckpt_file_name)
    label_file_path = os.path.join(LABEL_FILE_DIR, label_file_name)

    # load file from

    print('Loading tensorflow graph')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_file_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    print('Loading label file')

    label_map = label_map_util.load_labelmap(label_file_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=max_num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image_path_format = os.path.join(IMAGE_DIR, image_name_format)

    results = {}

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_index in range(num_image):
                image_path = image_path_format.format(image_index)
                image = Image.open(image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                result = create_image_detection_record(boxes[0], scores[0], classes[0], num[0])
                results[IMAGE_NAME_FORMAT.format(image_index)] = result
                # print(image_path, ' result:')
                # print(type(boxes))
                # print(boxes.shape)
                # print(boxes)
                # print(scores)
                # print(classes)
                # print(num)
                print("progress check: " + image_path + " Done. total " + str(num_image))
            np.save(os.path.join(RESULT_DIR, DETECTION_RESULT_FILE_NAME), results)
    return results


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def create_image_detection_record(boxes, scores, classes, num):
    record = []
    for i in range(int(num)):
        record.append({'box': boxes[i], 'score': scores[i], 'class': classes[i]})
    sorted(record, key=lambda e: e['score'], reverse=True)
    return record
