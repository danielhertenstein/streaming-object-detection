import cv2
import multiprocessing
import numpy as np
import os
import tensorflow as tf

from classes import WebcamWideoStream
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def capture(out_queue):
    # Setup
    stream = WebcamWideoStream(src=0).start()

    # Processing loop
    counter = 0
    while counter < 200:
        if out_queue.empty():
            frame = stream.read()
            out_queue.put(frame)
            counter += 1

    # Teardown
    stream.stop()


def detect(in_queue, out_queue):
    # Setup
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
        sess = tf.Session(graph=detection_graph)

    tensor_dict = {
        'num_detections': detection_graph.get_tensor_by_name('num_detections:0'),
        'detection_boxes': detection_graph.get_tensor_by_name('detection_boxes:0'),
        'detection_scores': detection_graph.get_tensor_by_name('detection_scores:0'),
        'detection_classes': detection_graph.get_tensor_by_name('detection_classes:0')
    }
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Processing loop
    counter = 0
    while counter < 200:
        frame = in_queue.get()
        expanded_frame = np.expand_dims(frame, 0)
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: expanded_frame})
        output_dict['frame'] = frame
        out_queue.put(output_dict)
        counter += 1


def display(in_queue):
    # Setup
    # Get labels and categories
    path_to_labels = os.path.join(os.getcwd(), 'object_detection', 'data', 'mscoco_label_map.pbtxt')
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=90,
        use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)

    # Processing loop
    counter = 0
    while counter < 200:
        output_dict = in_queue.get()
        frame = output_dict['frame']

        # Mark up the frame with the boxes
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output_dict['detection_boxes'][0],
            output_dict['detection_classes'][0].astype(np.uint8),
            output_dict['detection_scores'][0],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
        )

        # Show the marked up frame
        cv2.imshow("Frame", frame)
        cv2.waitKey(1) & 0xFF
        counter += 1

    # Teardown
    cv2.destroyAllWindows()


def main():
    capture_to_detect = multiprocessing.Queue(maxsize=1)
    detect_to_display = multiprocessing.Queue(maxsize=1)
    capturer = multiprocessing.Process(target=capture, args=(capture_to_detect,))
    detector = multiprocessing.Process(target=detect, args=(capture_to_detect, detect_to_display))
    detector.start()
    capturer.start()
    display(detect_to_display)


if __name__ == '__main__':
    main()


