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
pipeline = assembler.Pipeline(pipeline_pieces)

# Run the pipeline
pipeline.run()
