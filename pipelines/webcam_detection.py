import assembler
from sources import Webcam
from nodes import Detect, BoxMarkup
from sinks import Display

# Define the pieces that go in each process.
pipeline_pieces = [
    [Webcam],
    [Detect, BoxMarkup],
    [Display]
]

# Make the pipeline
pipeline = assembler.make_pipeline(pipeline_pieces)

# Start the pipeline
pipeline.start()

# Enter event loop and pass any messages to the sink

# Signal to the other pipeline pieces to stop

# Empty out all the queues

# Join the queues