from classes import WebcamVideoStream


class Source:
    def __init__(self, out_queue, keep_running):
        self.out_queue = out_queue
        self.keep_running = keep_running

    def setup(self):
        pass

    def process(self):
        pass

    def teardown(self):
        pass


class Webcam(Source):
    def __init__(self, out_queue, keep_running):
        super(Webcam, self).__init__(out_queue, keep_running)
        self.stream = WebcamVideoStream(src=0)

    def setup(self):
        self.stream.start()

    def process(self):
        while self.keep_running.value is True:
            if self.out_queue.empty():
                frame = self.stream.read()
                self.out_queue.put(frame)

    def teardown(self):
        self.stream.stop()
