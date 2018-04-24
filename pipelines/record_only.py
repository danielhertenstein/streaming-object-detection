import assembler
from pieces import WebcamCapture, Record


pipeline = assembler.Pipeline([
    [WebcamCapture,],
    [Record,],
])

pipeline.run()