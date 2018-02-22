import cv2
import numpy as np
import os
import tensorflow as tf

from classes import FPS, WebcamWideoStream
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util


CWD = os.getcwd()
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_CKPT = os.path.join(CWD, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
NUM_FRAMES = 100

# Get labels and categories
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=NUM_CLASSES,
    use_display_name=True
)
category_index = label_map_util.create_category_index(categories)

# Load model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)

# Single threaded
def single_threaded():
    print("[INFO] sampling frames from webcam")
    stream = cv2.VideoCapture(0)
    fps = FPS().start()

    while fps._numFrames < 100:
        # Get a new frame
        (grabbed, frame) = stream.read()

        # Do the detection
        ops = detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        keys = ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']
        for key in keys:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)

        if 'detection_masks' in tensor_dict:
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks,
                detection_boxes,
                frame.shape[0],
                frame.shape[1]
            )
            detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(frame, 0)})

        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]

        # Mark up the frame with the boxes
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8,
        )

        # Show the marked up frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    stream.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    single_threaded()

## Multi-threaded
#print("[INFO] sampling THREADED frames from webcam")
#stream = WebcamWideoStream(src=0).start()
#fps = FPS().start()
#
#while fps._numFrames < 100:
#    frame = stream.read()
#    output_dict = run_inference_for_single_image(frame, detection_graph)
#    vis_util.visualize_boxes_and_labels_on_image_array(
#        frame,
#        output_dict['detection_boxes'],
#        output_dict['detection_classes'],
#        output_dict['detection_scores'],
#        category_index,
#        instance_masks=output_dict.get('detection_masks'),
#        use_normalized_coordinates=True,
#        line_thickness=8,
#    )
#    cv2.imshow("Frame", frame)
#    key = cv2.waitKey(1) & 0xFF
#    fps.update()
#
#fps.stop()
#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#
#stream.stop()
#cv2.destroyAllWindows()
