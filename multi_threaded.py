import cv2
from multiprocessing import Pool, Queue
from multiprocessing.queues import Empty
import numpy as np
import os
import tensorflow as tf

from classes import FPS, WebcamWideoStream
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def object_detector(input_queue, output_queue):
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

    while True:
        frame = input_queue.get()
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: frame})
        output_queue.put(output_dict)


def main():
    # Get labels and categories
    path_to_labels = os.path.join(os.getcwd(), 'object_detection', 'data', 'mscoco_label_map.pbtxt')
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=90,
        use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)

    print("[INFO] sampling THREADED frames from webcam")
    input_queue = Queue(maxsize=1)
    output_queue = Queue(maxsize=1)
    pool = Pool(1, object_detector, (input_queue, output_queue))

    stream = WebcamWideoStream(src=0).start()
    fps = FPS().start()

    while fps.num_frames < 200:
        # Get a new frame
        frame = stream.read()
        expanded_frame = np.expand_dims(frame, 0)

        # Give it to the detector process
        if input_queue.empty():
            input_queue.put(expanded_frame)

        # Get the detection results
        try:
            output_dict = output_queue.get_nowait()
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            # Mark up the frame with the boxes
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
            )

            # Show the marked up frame
            cv2.imshow("Frame", frame)
            cv2.waitKey(1) & 0xFF

            fps.update()
        except Empty:
            pass

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    pool.terminate()
    stream.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
