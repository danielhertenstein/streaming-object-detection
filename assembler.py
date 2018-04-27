from collections import deque
from ctypes import c_bool
import multiprocessing
import queue


class Source:
    def __init__(self, pieces, options, keep_running, out_queue):
        self.pieces = [piece(**kwargs) for piece, kwargs in zip(pieces, options)]
        self.keep_running = keep_running
        self.out_queue = out_queue

    def main_loop(self):
        for piece in self.pieces:
            piece.setup()

        while self.keep_running.value is True:
            if self.out_queue.empty():
                data = None
                for piece in self.pieces:
                    data = piece.process(data)
                self.out_queue.put(data)

        for piece in self.pieces:
            piece.teardown()


class Node:
    def __init__(self, pieces, options, keep_running, in_queue, out_queue):
        self.pieces = [piece(**kwargs) for piece, kwargs in zip(pieces, options)]
        self.keep_running = keep_running
        self.in_queue = in_queue
        self.out_queue = out_queue

    def main_loop(self):
        for piece in self.pieces:
            piece.setup()

        while self.keep_running.value is True:
            data = self.in_queue.get()
            for piece in self.pieces:
                data = piece.process(data)
            self.out_queue.put(data)

        for piece in self.pieces:
            piece.teardown()


class Sink:
    def __init__(self, pieces, options, in_queue):
        self.pieces = [piece(**kwargs) for piece, kwargs in zip(pieces, options)]
        self.in_queue = in_queue

    def main_loop(self):
        for piece in self.pieces:
            piece.setup()

        while True:
            data = self.in_queue.get()
            for piece in self.pieces:
                data = piece.process(data)
            if data is False:
                break

        for piece in self.pieces:
            piece.teardown()


class Pipeline:
    def __init__(self, pieces, options):
        self.keep_running = multiprocessing.Value(c_bool, True)
        self.queues = deque(multiprocessing.Queue(maxsize=1) for _ in range(len(pieces)- 1))
        self.pieces = []

        # Make the source
        out_queue = self.queues[0]
        piece = Source(pieces.pop(0), options.pop(0), self.keep_running, out_queue)
        self.pieces.append(multiprocessing.Process(target=piece.main_loop))
        self.queues.rotate(-1)

        # Make any intermediary pieces
        while len(pieces) > 1:
            in_queue = out_queue
            out_queue = self.queues[0]
            piece = Node(pieces.pop(0), options.pop(0), self.keep_running, in_queue, out_queue)
            self.pieces.append(multiprocessing.Process(target=piece.main_loop))
            self.queues.rotate(-1)

        # Make the sink
        in_queue = out_queue
        self.sink = Sink(pieces.pop(0), options.pop(0), in_queue)

    def run(self):
        # Start all of the non-sink processes
        for piece in self.pieces:
            piece.start()

        # Run the sink method
        self.sink.main_loop()

        # Signal the pieces to stop
        with self.keep_running.get_lock():
            self.keep_running.value = False

        # Clear out all of the queues
        queues_to_clear = len(self.queues)
        while queues_to_clear > 0:
            for node_queue in self.queues:
                try:
                    node_queue.get_nowait()
                    queues_to_clear -= 1
                except queue.Empty:
                    pass
                self.keep_running.value = False

        # Join the pipeline processes
        for piece in self.pieces:
            piece.join()
