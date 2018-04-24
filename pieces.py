import cv2
import numpy as np
import os
import time
import tensorflow as tf

from classes import WebcamVideoStream
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class Markup:
    def __init__(self):
        self.category_index = None

    def setup(self):
        # Get labels and categories
        path_to_labels = os.path.join(os.getcwd(), 'object_detection', 'data', 'mscoco_label_map.pbtxt')
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=90,
            use_display_name=True
        )
        self.category_index = label_map_util.create_category_index(categories)

    def process(self, data):
        frame = data['frame']

        # Mark up the frame with the boxes
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            data['detection_boxes'][0],
            data['detection_classes'][0].astype(np.uint8),
            data['detection_scores'][0],
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
        )
        return frame

    def teardown(self):
        pass


class Display:
    def setup(self):
        pass

    def process(self, data):
        # Show the marked up frame
        cv2.imshow("Frame", data)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False

    def teardown(self):
        cv2.destroyAllWindows()


class Record:
    def __init__(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.video_writer = cv2.VideoWriter('{}.avi'.format(timestamp), fourcc, 20.0, (640, 480))

    def setup(self):
        pass

    def process(self, data):
        self.video_writer.write(data)
        return True

    def teardown(self):
        self.video_writer.release()


class Detect:
    def __init__(self):
        self.sess = None
        self.tensor_dict = None
        self.image_tensor = None

    def setup(self):
        cwd = os.getcwd()
        model_name = 'ssd_mobilenet_v2_coco_2018_03_29'
        path_to_ckpt = os.path.join(cwd, 'object_detection', model_name, 'frozen_inference_graph.pb')

        # Load model into memory
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=detection_graph)

        self.tensor_dict = {
            'num_detections': detection_graph.get_tensor_by_name('num_detections:0'),
            'detection_boxes': detection_graph.get_tensor_by_name('detection_boxes:0'),
            'detection_scores': detection_graph.get_tensor_by_name('detection_scores:0'),
            'detection_classes': detection_graph.get_tensor_by_name('detection_classes:0')
        }
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    def process(self, data):
        expanded_frame = np.expand_dims(data, 0)
        output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: expanded_frame})
        output_dict['frame'] = data
        return output_dict

    def teardown(self):
        pass


class WebcamCapture:
    def __init__(self):
        self.stream = None

    def setup(self):
        self.stream = WebcamVideoStream(src=0).start()

    def process(self, data):
        return self.stream.read()

    def teardown(self):
        self.stream.stop()


#class PiCapture:
#    def __init__(self):
#        self.stream = None
#
#    def setup(self):
#        self.stream = PiVideoStream(src=0).start()
#
#    def process(self, data):
#        return self.stream.read()
#
#    def teardown(self):
#        self.stream.stop()
