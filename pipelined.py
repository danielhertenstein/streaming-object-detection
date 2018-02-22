from ctypes import c_bool
import cv2
import multiprocessing
import numpy as np
import os
import tensorflow as tf

from classes import WebcamWideoStream
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def capture(keep_running, out_queue):
    # Setup
    stream = WebcamWideoStream(src=0).start()

    # Processing loop
    while keep_running.value is True:
        if out_queue.empty():
            frame = stream.read()
            out_queue.put(frame)

    # Teardown
    stream.stop()


def detect(keep_running, in_queue, out_queue):
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
    while keep_running.value is True:
        frame = in_queue.get()
        expanded_frame = np.expand_dims(frame, 0)
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: expanded_frame})
        output_dict['frame'] = frame
        out_queue.put(output_dict)


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
    while True:
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Teardown
    cv2.destroyAllWindows()


def main():
    # Setup the queues
    capture_to_detect = multiprocessing.Queue(maxsize=1)
    detect_to_display = multiprocessing.Queue(maxsize=1)
    # Setup the shared exit trigger
    keep_running = multiprocessing.Value(c_bool, True)
    # Create the pipeline pieces
    capturer = multiprocessing.Process(target=capture, args=(keep_running, capture_to_detect,))
    detector = multiprocessing.Process(target=detect, args=(keep_running, capture_to_detect, detect_to_display))
    # Start the pipeline pieces
    detector.start()
    capturer.start()
    # Run the "GUI"
    display(detect_to_display)
    # Signal the pipeline pieces to stop
    with keep_running.get_lock():
        keep_running.value = False
    # Clear out the queues
    detect_to_display.get()
    capture_to_detect.get()
    # Join the pipeline processes
    capturer.join()
    detector.join()


if __name__ == '__main__':
    main()
