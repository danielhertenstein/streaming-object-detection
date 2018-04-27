import assembler
from pieces import WebcamCapture, Record


def main():
    pipeline = assembler.Pipeline(
        pieces=[[WebcamCapture], [Record]],
        options=[[{}], [{'framerate': 120.0}]]
    )
    pipeline.run()


if __name__ == '__main__':
    main()
