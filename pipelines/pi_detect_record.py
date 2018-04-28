import assembler
from pieces import PiCapture, Record, Detect, Markup


def main():
    pipeline = assembler.Pipeline(
        pieces=[[PiCapture], [Detect], [Markup, Record]],
        options=[[{}], [{}], [{}, {'framerate': 1.0}]]
    )
    pipeline.run()


if __name__ == '__main__':
    main()
