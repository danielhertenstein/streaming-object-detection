import cv2
import numpy as np
import os
import tensorflow as tf

from classes import FPS
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


CWD = os.getcwd()
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = os.path.join(CWD, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Get labels and categories
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=NUM_CLASSES,
    use_display_name=True
)
category_index = label_map_util.create_category_index(categories)


def main():
    # Load model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    print("[INFO] sampling frames from webcam")
    stream = cv2.VideoCapture(0)
    fps = FPS().start()

    while fps.num_frames < 200:
        # Get a new frame
        (grabbed, frame) = stream.read()

        # Do the detection
        tensor_dict = {
            'num_detections': detection_graph.get_tensor_by_name('num_detections:0'),
            'detection_boxes': detection_graph.get_tensor_by_name('detection_boxes:0'),
            'detection_scores': detection_graph.get_tensor_by_name('detection_scores:0'),
            'detection_classes': detection_graph.get_tensor_by_name('detection_classes:0')
        }
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        expanded_frame = np.expand_dims(frame, 0)
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: expanded_frame})

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

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    stream.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
