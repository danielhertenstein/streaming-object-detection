from ctypes import c_bool
import multiprocessing


class Pipeline:
    def __init__(self, pipeline_pieces):
        self.queues = [multiprocessing.Queue(maxsize=1) for _ in range(len(pipeline_pieces))]
        self.keep_running = multiprocessing.Value(c_bool, True)

        self.pieces = []
        # Make the source
        queue = self.queues[0]
        pieces = pipeline_pieces[0]
        self.pieces.append(self.make_process(pieces, out_queue=queue))
        # Loop over all but last pieces
        for i, pieces in enumerate(pipeline_pieces[1:-1]):
            in_queue = self.queues[i-1]
            out_queue = self.queues[i]
            self.pieces.append(self.make_process(pieces, in_queue=in_queue, out_queue=out_queue))
        # Make the sink
        queue = self.queues[-1]
        pieces = pipeline_pieces[-1]
        self.sink = self.make_process(pieces, in_queue=queue)

    def make_process(self, pieces, in_queue=None, out_queue=None):
        return "process"

    def run(self):
        for piece in self.pieces:
            piece.start()

        self.sink.process()

        with self.keep_running.get_lock():
            self.keep_running.value = False

        # TODO: Cycle through all queues until all clear
        for queue in self.queues:
            queue.get()

        for piece in self.pieces:
            piece.join()
