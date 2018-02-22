import cv2

from classes import FPS, WebcamWideoStream


NUM_FRAMES = 100


# Single threaded
print("[INFO] sampling frames from webcam")
stream = cv2.VideoCapture(0)
fps = FPS().start()

while fps._numFrames < 100:
    (grabbed, frame) = stream.read()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

stream.release()
cv2.destroyAllWindows()

# Multi-threaded
print("[INFO] sampling THREADED frames from webcam")
stream = WebcamWideoStream(src=0).start()
fps = FPS().start()

while fps._numFrames < 100:
    frame = stream.read()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

stream.stop()
cv2.destroyAllWindows()
