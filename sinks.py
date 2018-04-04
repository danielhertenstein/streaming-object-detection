import cv2


class Sink:
    def __init__(self, in_queue):
        self.in_queue = in_queue

    def setup(self):
        pass

    def process(self):
        pass

    def teardown(self):
        pass


class Display(Sink):
    def __init__(self, in_queue):
        super(Display, self).__init__(in_queue)

    def process(self):
        while True:
            frame = self.in_queue.get()

            # Show the marked up frame
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def teardown(self):
        cv2.destroyAllWindows()
