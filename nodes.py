import numpy as np
import os
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class Node:
    def __init__(self, in_queue, out_queue, keep_running):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.keep_running = keep_running

    def setup(self):
        pass

    def process(self):
        pass

    def teardown(self):
        pass


class Detect(Node):
    def __init__(self, in_queue, out_queue, keep_running):
        super(Detect, self).__init__(in_queue, out_queue, keep_running)
        self.sess = None
        self.tensor_dict = None
        self.image_tensor = None

    def setup(self):
        cwd = os.getcwd()
        model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
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

    def process(self):
        while self.keep_running.value is True:
            frame = self.in_queue.get()
            expanded_frame = np.expand_dims(frame, 0)
            output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: expanded_frame})
            output_dict['frame'] = frame
            self.out_queue.put(output_dict)


class BoxMarkup(Node):
    def __init__(self, in_queue, out_queue, keep_running):
        super(BoxMarkup, self).__init__(in_queue, out_queue, keep_running)
        self.category_index = None

    def setup(self):
        path_to_labels = os.path.join(os.getcwd(), 'object_detection', 'data', 'mscoco_label_map.pbtxt')
        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=90,
            use_display_name=True
        )
        self.category_index = label_map_util.create_category_index(categories)

    def process(self):
        while self.keep_running.value is True:
            output_dict = self.in_queue.get()
            frame = output_dict['frame']

            # Mark up the frame with the boxes
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                output_dict['detection_boxes'][0],
                output_dict['detection_classes'][0].astype(np.uint8),
                output_dict['detection_scores'][0],
                self.category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
            )

            # Put the frame in the out queue
            self.out_queue.put(frame)
