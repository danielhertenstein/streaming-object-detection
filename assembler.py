class Pipeline:
    def __init__(self):
        self.pieces = None

    def start(self):
        for piece in self.pieces:
            piece.start()


def make_process(pieces):
    pass


def make_pipeline(pipeline_pieces):
    return "pipeline"